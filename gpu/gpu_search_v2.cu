#include "gpu_search_v2.hh"
#include "gpu_common.h"
#include "memory_management.hh"

#define big_entry 16
#define small_entry 8

struct identity {
    __device__ void operator()(float& num) {
        return;
    }
};

struct exponentify {
    __device__ void operator()(float& num) {
        num = expf(num);
    }
};

template<unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram>
__device__ void traversal (int count, int track, uint64_t updated_index, unsigned int size, unsigned char * btree_trie_mem, float * results) {
printf("This is thread %d's traversal for %d time \n",track, count);
    count++;
    unsigned int offsets[max_num_children/2 +1];
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];
    unsigned int fetch_entries[entries_per_node+1];
    bool is_last_lvl=false;

    int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(big_entry + sizeof(unsigned short));
    is_last_lvl = !(entries_per_node == cur_node_entries);
    
    uint64_t sub_idx[entries_per_node];
    unsigned int sub_size[entries_per_node];
    
    if (is_last_lvl) {
        int num_entries = size/big_entry; 
        printf("is_last_lvl: num_entries%d; track is %d\n",num_entries,track);
        for (int i = 0; i < num_entries ; i++) {
            fetch_entries[i] = *(unsigned int *)(&btree_trie_mem[updated_index + i*sizeof(unsigned int)]);    
            //printf("testprob%f\n",*prob);
            unsigned int tmp;
            tmp=(*(unsigned int *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*(sizeof(unsigned int) + sizeof(float) + sizeof(float))  + sizeof(unsigned int)])); 
//*(float *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*(sizeof(float))]);          
            float *tmp_prob=(float *)&tmp;
            printf("traversal, thread is %d, loop is , prob is%f\n",track, i,*tmp_prob);
            printf("entries associated with above line%d,%d\n",i,fetch_entries[i]);
    }
} else {
        int num_entries = entries_per_node;
        for (int i = 0; i < entries_per_node ; i++) {
            unsigned int tmp;
            fetch_entries[i] = *(unsigned int *)(&btree_trie_mem[updated_index + i*sizeof(unsigned int)]);
            tmp=*(unsigned int *)(&btree_trie_mem[updated_index + sizeof(unsigned int) + max_num_children*sizeof(unsigned short)  + num_entries*sizeof(unsigned int)  + i*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) + 1*sizeof(unsigned int)]);
            float *tmp_prob=(float *)&tmp;
            printf("no last traversal thread is %d\n",track);
            printf("loop is ,%d, prob is %f\n",i,*tmp_prob);
            printf("entries associated with above line:%d\n",fetch_entries[i]);

            if (i < (max_num_children/2) + 1) {
                offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
				unsigned int test = *(unsigned int *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + 1*sizeof(unsigned int)]);
				printf("test %d\n",test);
				printf("thread :%d - offsets[%d] is %d: \n",track, i, offsets[i]);
            }
        }
        for(int i = 0; i < max_num_children ; i++) {
    //        uint64_t sub_idx[entries_per_node];
      //      unsigned int sub_size[entries_per_node];
            sub_idx[0]=updated_index + *first_child_offset*4;
            if (i == 0) { 
            sub_size[i] = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it 
            } else {
            sub_idx[i]=sub_idx[0] + offests_incremental[i - 1]*4;
            sub_size[i] = (offests_incremental[i] - offests_incremental[i - 1])*4;
            }
            
            
            if (sub_size[i]!=0){
			printf("also begin ith traversal. test sub_size[%d] is %d\n",i, sub_size[i]);
            traversal<max_num_children, entries_per_node, max_ngram>(count, track, sub_idx[i],sub_size[i], btree_trie_mem, results);
            }
        }
    }
}


template<unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram, class Functor>
__global__ void gpuSearchBtree(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, float * results, Functor fn) {

    __shared__ unsigned int offsets[max_num_children/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries_actual[entries_per_node + 1];
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[max_ngram]; //Each block fetches from shared memory the max necessary number of keys

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < max_ngram) {
       keys_shared[i] = keys[(blockIdx.x*max_ngram) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    if (i == 0) {
        //Initialize shared memory for search. We write the entries (keys) from position 1 to n and put
        //0 at position 0 of the actual array. This allows us to skip a case when doing an nary search.
        //Potentially we could set the entries_actual[num_entries +1] element to UINT_MAX and then by
        //using an extra thread skip another divergence case (which will be moved to the memory copy part)
        //Not sure if it's worth it cause it requires a rewrite of the btree part
        entries_actual[0] = 0;
    }
    __syncthreads();

    unsigned int * entries = &entries_actual[1];

    //Setup global memory for convenience
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];

    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];

    unsigned int * next_level = &payload[0];
    float * prob = (float *)&payload[1];
    float * backoff = (float *)&payload[2];

    //Backoff variables
    unsigned int match_length_found = 0; //To check what was our longest match so we know what to backoff to
    float accumulated_score = 0;
    bool get_backoff = false; //Are we looking to extract backoff or prob from our ngram

    //First get the value from first_lvl
    unsigned int current_ngram = 0;
    unsigned int key = keys_shared[current_ngram];
	
	uint64_t btree_start;
    uint64_t updated_index;
    unsigned int btree_size;
    /* When using gLM to score ngrams for NMT frequently sentences in batches are padded with zeroes so they can be at the same length
    * the easiest way to get corresponding behaviour is to allow gLM to submit bogus scores (e.g. 0) for them in case the first ngram
    * in the query is zero. Hence this ugly goto which will bypass the btree code. Unfortunately we pay for this with about 0.001% drop
    * in throughput ;/
    *//*
	printf("Test the content in key ------------------\n");
	printf ("%d\n",keys_shared[0]);
	printf ("%d\n",keys_shared[1]);
	printf ("%d\n",keys_shared[2]);
printf ("%d\n",keys_shared[3]);
printf ("%d\n",keys_shared[4]);*/
    if (key != 0) {

        //Backoff logic
        backoff_part2:
        if (get_backoff) {
            accumulated_score += *prob; //We add the longest match of probability we found.
            match_length_found = current_ngram - 1; //The length of the match found. We need to backoff from toplevel to here
            current_ngram = 1; //Set backoff in -1. If we run into this case again we need to do nothing
            key = keys_shared[current_ngram];
            get_backoff = true;
        }
        __syncthreads(); //Needed!
        if (i < 3) {
            payload[i] = first_lvl[(key-1)*3 + i];
            //If this is the last non zero ngram  no need to go to the btree_trie. We already have
            //the payload value
        }
        __syncthreads();
        if (i == 0) {
            if (get_backoff && match_length_found <= current_ngram) {
                accumulated_score += *backoff;
            } else if (keys_shared[current_ngram + 1] == 0) {
                accumulated_score += *prob;
            }
        }
        __syncthreads();

        //Set the start index
        uint64_t current_btree_start = *next_level*4;
		printf("111testcurrent_btree_start%d\n",current_btree_start);
		btree_start = current_btree_start;
        current_ngram++;
        key = keys_shared[current_ngram];

        //Some necessary variables
        uint64_t updated_idx;
        unsigned int size;

        //Current_btree_start == 0 means that we had UNK key (vocabID 1 which wasn't found, so we should directly go to backoff
        //@TODO we can check if key[0] == 1 when we get the score too
        if (current_btree_start == 0 && key != 0) {
            goto backoff_notriecont;
        }

        while ((key != 0 && current_ngram < max_ngram - 1 && current_btree_start != 0) || 
            (get_backoff && key != 0 && current_ngram < max_ngram && current_btree_start != 0)) {
            current_ngram++;
			printf("current_ngram%d\n",current_ngram);
            updated_idx = current_btree_start + 4; //Update the index for the while loop
			printf("testcurrentstartsame129 is %d\n",current_btree_start);
            //@TODO consider this for shared memory as oppposed to global mem broadcast to register
            size = *(unsigned int *)&btree_trie_mem[current_btree_start]; //The size of the current node to process.
			printf("46288sizeis %d\n",size);
            //Initialize shared variable
            if (i < 2) {
                booleans[i] = false; //Uset *exact_match and *is_last
            }
            __syncthreads();

            //Traverse the current Btree
            while (!*exact_match) {
                //First warp divergence here. We are reading in from global memory
                if (i == 0) {
                    //@TODO: Replace this with a mod check
                    int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(big_entry + sizeof(unsigned short));
                    *is_last = !(entries_per_node == cur_node_entries);
                }
                __syncthreads();

                int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.
                if (*is_last) {
                    //The number of entries in the bottom most nodes may be smaller than the size
                    num_entries = size/big_entry;
					printf("heis152num_entries%d\n",num_entries);
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                } else {
                    num_entries = entries_per_node;
                    //Now load the entries
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }

                    //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                    if (i < (max_num_children/2) + 1) {
                        offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
						printf("test <s> in ----------%d\n",offsets[i]);
						printf("test why it run 5 times:thread%d,content: %d\n",i,offsets[i]);
                    }
                }
                __syncthreads();

                //NOW search
                if (i < num_entries) {
                    if (key > entries_actual[i] && key <= entries_actual[i + 1]){
                        found_idx = i;
                        if (key == entries_actual[i + 1]) {
                            *exact_match = true;
                        }
                    }
                } else if (i == num_entries) {
                    //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                    if (key > entries_actual[i]) {
                        found_idx = i;
                    }
                }
                __syncthreads();

                //We found either an exact match (so we can access next level) or at least an address to next btree level
                if (!*exact_match && !*is_last) {
                    //Calculate the address and the size of the next child
                    updated_idx += *first_child_offset*4;
                    if (found_idx == 0) {
                       size = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it
                    } else {
						//printf("249could i > 4%d",offests_incremental[6]);
                        updated_idx += offests_incremental[found_idx - 1]*4;
                        size = (offests_incremental[found_idx] - offests_incremental[found_idx - 1])*4;
                    }
                    __syncthreads();
                } else if (*is_last && !*exact_match) {
                    //In this case we didn't find the key that we were looking for
                    //What we should do is get the probability of the last node that we found
                    //The last node that we found's probability should be in shared memory
                    backoff_notriecont:
                    if (get_backoff) {
                        current_ngram = max_ngram;
                        break; //If we didn't find a backoff, the value is zero; //We should go to end now, because any further backoffs
                        // will be missing from the trie
                    } else {
                        get_backoff = true;
                        __syncthreads(); //Necessary
                        goto backoff_part2;
                    }
                } else {
                    //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
                    if (i < 3) {
                        //What we are doing here is reading the correct memory location for our payload. The payload is found
                        //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                        if (*is_last) {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                    + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                        } else {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                                + num_entries*sizeof(unsigned int) //Skip the keys
                                    + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                        + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                        }
                    }

                    key = keys_shared[current_ngram]; //@TODO this might be illegal memory access
                    __syncthreads();

                    current_btree_start = current_btree_start + *next_level*4;
					btree_start = current_btree_start;//+ *next_level*4;
					printf("234current_btree_start%d\n",current_btree_start);
					printf("235btree_start%d\n",btree_start);
                    if (current_btree_start == btree_start){printf("testcurrent344%d,%d\n",btree_start,current_btree_start);
}
                    //Very rarely, mostly when having big datasets with small vocabulary
                    //we will have tries that don't go to the last level. In this case
                    //we just need to initiate backoff
                    if (*next_level == 0 && key != 0) {
                        current_ngram++; //We need to add one to the current_ngram because we actually found a match on this trie level
                        goto backoff_notriecont; //it's the next trie level we are missing so in effect we say that this is the longest
                    }                            //match and we need to calculate the backoff for the rest, similar to the case in the last_level


                    if (get_backoff) {
                        if (match_length_found < current_ngram) {
                            accumulated_score += *backoff;
                        }
                    } else if (key == 0) {
                        accumulated_score += *prob;
                    }

                    break;
                }
            }
        }
        //Now fetch the last level if the key is not 0 or we backed off
        //key = keys_shared[current_ngram]; We already set the next key
        if (!get_backoff && key != 0) {
			printf(">?????");
            updated_idx = current_btree_start + 4; //Update the index for the while loop
            //@TODO consider this for shared memory as oppposed to global mem broadcast to register
            size = *(unsigned int *)&btree_trie_mem[current_btree_start]; //The size of the current node to process.

            //Initialize shared variable
            if (i < 2) {
                booleans[i] = false;
            }
            __syncthreads();

            //Traverse the current Btree
            while (!*exact_match) {
                //First warp divergence here. We are reading in from global memory
                if (i == 0) {
                    //@TODO: Replace this with a mod check
                    int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(small_entry + sizeof(unsigned short));
                    *is_last = !(entries_per_node == cur_node_entries);
                }
                __syncthreads();

                int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.
                if (*is_last) {
                    //The number of entries in the bottom most nodes may be smaller than the size
                    num_entries = size/small_entry;
                    printf("num_entries279%d\n",num_entries);
					if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                } else {
                    num_entries = entries_per_node;
                    //Now load the entries
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                    //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                    if (i < (max_num_children/2) + 1) {
                        offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
                    }
                }
                __syncthreads();

                //NOW search
                if (i < num_entries) {
                    if (key > entries_actual[i] && key <= entries_actual[i + 1]){
                        found_idx = i;
                        if (key == entries_actual[i + 1]) {
                            *exact_match = true;
                        }
                    }
                } else if (i == num_entries) {
                    //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                    if (key > entries_actual[i]) {
                        found_idx = i;
                    }
                }
                __syncthreads();

                //We found either an exact match (so we can access next level) or at least an address to next btree level
                if (!*exact_match && !*is_last) {
                    //Calculate the address and the size of the next child
                    updated_idx += *first_child_offset*4;
                    if (found_idx == 0) {
                       size = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it
                    } else {
                        updated_idx += offests_incremental[found_idx - 1]*4;
                        size = (offests_incremental[found_idx] - offests_incremental[found_idx - 1])*4;
                    }
                    __syncthreads();
                } else if (!*exact_match && is_last) {
                    current_ngram++; //This is necessary so that longest match logic is kept correct since in the while loop we
                    goto backoff_notriecont; //Increment this before actually finding the next level
                } else {
                    // We have an exact match, so we just need to add it to the payload and be done with it
                    if (i == 3) {
						if (*is_last) {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                    + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                        } else {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                                + num_entries*sizeof(unsigned int) //Skip the keys
                                    + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                        + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                        }
                    }
					key = keys_shared[current_ngram+1];
					__syncthreads();
                    btree_start = current_btree_start + *next_level*4;
				//	printf("testcurrent344%d",current_btree_start);
                }
            }
        }

    } //key != 0

	printf("LgLMbegincurrent_ngramtest%d",current_ngram);//if it is already max_ngram I have to change the way to calculate num_entries and the way to get prob etc, but others should be the same because the last one is different from the previous (no children this is the ngram level not a Btree[eg:ngram A,B,C,D,E for E it has no children so it is different to get data for  E from C, but E still has so many vocabs can be])
	unsigned int fetch_entries[entries_per_node+1];
	float  score[entries_per_node + 1];
unsigned int next_address[entries_per_node +1];
//printf ("key is %d\n",key);
//printf ("%d\n",keys_shared[0]);
//printf ("%d\n",keys_shared[1]);
//printf ("%d\n",keys_shared[2]);
//printf ("%d\n",keys_shared[3]);
//printf ("%d\n",current_ngram);
if (key == 0&&current_ngram <max_ngram){//if it is =max_ngram then I have to change to a new way
	uint64_t sub_idx[entries_per_node];
    unsigned int sub_size[entries_per_node];
    updated_index = btree_start + 4;
    btree_size = *(unsigned int *)&btree_trie_mem[btree_start];
    if (i < 2) {
        booleans[i] = false;
    }
    __syncthreads();
    if (i == 0){
        int cur_node_entries = (btree_size - sizeof(unsigned int) - sizeof(unsigned short))/(big_entry + sizeof(unsigned short));
        *is_last = !(entries_per_node == cur_node_entries);
    }
    __syncthreads();
    int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.
    if (*is_last) {
    //The number of entries in the bottom most nodes may be smaller than the size
        num_entries = btree_size/big_entry;
        if (i < num_entries) {// all the three entries are found but the score is wrong
            fetch_entries[i] = *(unsigned int *)(&btree_trie_mem[updated_index + i*sizeof(unsigned int)]);
			unsigned int tmp;
            tmp=(*(unsigned int *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*(sizeof(unsigned int) + sizeof(float) + sizeof(float))  + sizeof(unsigned int)])); 
//*(float *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*(sizeof(float))]); 
			float *tmp_prob=(float *)&tmp;
			printf("testscore%d,%f\n",i,*tmp_prob);
			printf("testentries%d,%d\n",i,fetch_entries[i]);
        }
    } else {
	//	printf("test 7 is %d\n",*(unsigned int *)(&btree_trie_mem[updated_index + 7*sizeof(unsigned int)])); after test 7 is not the child of <s>
        num_entries = entries_per_node;
		printf("begin the no last travel and num_entries is %d\n",num_entries);
         //Now load the entries
        if (i < num_entries) {
				unsigned int tmp;
                fetch_entries[i] = *(unsigned int *)(&btree_trie_mem[updated_index + i*sizeof(unsigned int)]);
				
            //    next_address[i] = *(unsigned int *)(&btree_trie_mem[updated_index + sizeof(unsigned int) + max_num_children*sizeof(unsigned short)  + num_entries*sizeof(unsigned int)  + i*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) + 0*sizeof(unsigned int)]); 
				tmp=*(unsigned int *)(&btree_trie_mem[updated_index + sizeof(unsigned int) + max_num_children*sizeof(unsigned short)  + num_entries*sizeof(unsigned int)  + i*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) + 1*sizeof(unsigned int)]);  
				float *tmp_prob=(float *)&tmp;
				printf("testscoreelae,%d,%f\n",i,*tmp_prob);
	            printf("testentrieelse,%d\n",fetch_entries[i]);
			//	printf("testaddresselse,%d\n",next_address[i]);
            }
        if (i < (max_num_children/2) + 1) {
			printf("moniter thread : %d\n",i);
            offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_index + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
        }
			//uint64_t sub_idx[num_entries];
            //unsigned int sub_size[num_entries];
/*
            sub_idx[0]=updated_index + *first_child_offset*4;
			printf("sub_idx[0] is %d\n",sub_idx[0]);
            if (i == 0) {
            sub_size[i] = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it
            } else {
            sub_idx[i]=sub_idx[0] + offests_incremental[i - 1]*4;
            sub_size[i] = (offests_incremental[i] - offests_incremental[i - 1])*4;
			printf("test sub_idx[%d] is %d, and associated offests_incremental[%d] is %d\n",i,sub_idx[i],i,offests_incremental[i]);
            } */
			
        }
		__syncthreads();
		updated_index +=*first_child_offset*4;
		if (i == 0){
		sub_idx[0]=updated_index;//sub_idx[0]=140
        sub_size[i] = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode i
        } else {
        sub_idx[i]=updated_index + offests_incremental[i - 1]*4;
        sub_size[i] = (offests_incremental[i] - offests_incremental[i - 1])*4;
		}
		__syncthreads();
		printf("begin threadIdx %d travel",i);
		traversal<max_num_children,entries_per_node,max_ngram>(1, i, sub_idx[i],sub_size[i], btree_trie_mem, results);
//	    traversal(fetch_btree_start[i], max_num_children, entries_per_node, max_ngram, btree_trie_mem, results);
    }
	
    //Write the correct result at the end
    if (i == 0) {
        fn(accumulated_score); //This is basically either identity or exp, depending on what we need
        results[blockIdx.x] = accumulated_score;
    }
}


/*
    We have to do this to provide some degree of flexibility, whilst maintaining performance
    http://stackoverflow.com/questions/32534371/cuda-most-efficient-way-to-store-constants-that-need-to-be-parsed-as-arguments?noredirect=1#comment52933276_32534371
    http://stackoverflow.com/questions/6179295/if-statement-inside-a-cuda-kernel/6179580#6179580
    http://stackoverflow.com/questions/31569401/fastest-or-most-elegant-way-of-passing-constant-arguments-to-a-cuda-kernel?rq=1
    Instantiate templates for known things:
*/
inline void kernelTemplateWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_num_children,
  unsigned int max_ngram, cudaStream_t& stream, bool make_exp){
    if (max_ngram == 6) {
        if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 6><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 6><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else { 
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if (max_ngram == 5) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if (max_ngram == 4) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if ( max_ngram == 3) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());            
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else {
        printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
    }
}

inline void kernelTemplateWrapperDebug(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_num_children,
  unsigned int max_ngram, cudaStream_t& stream, cudaEvent_t &start, cudaEvent_t &stop, bool make_exp){
    cudaEventRecord(start);
    kernelTemplateWrapper(btree_trie_mem, first_lvl,  keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
}

void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, bool make_exp, bool debug) {

    unsigned int max_num_children = entries_per_node + 1;
    cudaStream_t stream;
    CHECK_CALL(cudaStreamCreate(&stream));

    if (debug) {
        //Time the kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        kernelTemplateWrapperDebug(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node,
         max_num_children, max_ngram, stream, start, stop, make_exp);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
        printf("Throughput: %d queries per second.\n", (int)((num_ngram_queries/(milliseconds))*1000));
    } else {
        kernelTemplateWrapper(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    }
    CHECK_CALL(cudaStreamDestroy(stream));
}

void searchWrapperStream(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, cudaStream_t& stream, bool make_exp, bool debug) {


    unsigned int max_num_children = entries_per_node + 1;

    if (debug) {
        //Time the kernel execution @TODO remove once its working
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        kernelTemplateWrapperDebug(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node,
         max_num_children, max_ngram, stream, start, stop, make_exp);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
        printf("Throughput: %d queries per second.\n", (int)((num_ngram_queries/(milliseconds))*1000));
    } else {
        kernelTemplateWrapper(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    }
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID) {
    CHECK_CALL(cudaSetDevice(deviceID));
}

void GPUSearcher::search(unsigned int * keys, unsigned int num_ngram_queries, float * results, int streamID, bool debug) {
    if (streamID > num_streams - 1) {
        std::cerr << "Provided stream greater than the available ones. Using stream 0 as default. Fix your code!" << std::endl;
        streamID = 0;
    }

    searchWrapperStream(btree_trie_gpu, first_lvl_gpu, keys, num_ngram_queries, results, lm.metadata.btree_node_size,
     lm.metadata.max_ngram_order, streams[streamID], make_exp, debug);
}

std::vector<float> GPUSearcher::search(std::vector<unsigned int>& queries, int streamID, bool debug) {
    if (streamID > num_streams - 1) {
        std::cerr << "Provided stream greater than the available ones. Using stream 0 as default. Fix your code!" << std::endl;
        streamID = 0;
    }

    unsigned int num_ngram_queries = queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_ngram_queries, &results);

    searchWrapperStream(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_ngram_queries, results, lm.metadata.btree_node_size,
     lm.metadata.max_ngram_order, streams[streamID], make_exp, debug);

    std::vector<float> cpuResults(num_ngram_queries);

    copyToHostMemory(results, cpuResults.data(), num_ngram_queries);

    //Free memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return cpuResults;
}

void GPUSearcher::gpuInit() {
    //Init GPU memory
    btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    if (num_streams < 1) {
        std::cerr << "You have specified " << num_streams << " number of streams however it must be at least 1. Using 1 stream as default. Fix your code!" << std::endl;
        num_streams = 1;
    }
    streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CHECK_CALL(cudaStreamCreate(&streams[i]));
    }
}

GPUSearcher::GPUSearcher(int num, LM& lm_, bool make_exp_) : lm(lm_), num_streams(num), make_exp(make_exp_) {
    gpuInit();
}

GPUSearcher::GPUSearcher(int num, LM& lm_, int gpuDeviceID, bool make_exp_) : lm(lm_), num_streams(num), make_exp(make_exp_) {
    setGPUDevice(gpuDeviceID);
    //Init GPU memory
    gpuInit();
}

GPUSearcher::~GPUSearcher() {
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    for (int i = 0; i < num_streams; i++) {
        CHECK_CALL(cudaStreamDestroy(streams[i]));
    }
}
