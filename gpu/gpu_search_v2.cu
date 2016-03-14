#include "gpu_search.hh" 
#include <cuda_runtime.h>
#include "gpu_common.h"

#define big_entry 16
#define small_entry 8
extern __shared__ unsigned int shared_mem[];
//template<unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram>
__global__ void gpuSearchBtree(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, float * results, unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram) {

    unsigned int * offsets = shared_mem; //Reads in the first child offset + the shorts
    unsigned int * entries = &shared_mem[max_num_children/2 +1];
    unsigned int * found_idx = &entries[entries_per_node];
    unsigned int * booleans = &found_idx[1]; //booleans[0] = is_last; booleans[1] = exact_match
    unsigned int * payload = &booleans[2]; //After we find the correct entry, load the payload here
    unsigned int * keys_shared = &payload[3]; //Each block fetches from shared memory the max necessary number of keys

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < max_ngram) {
       keys_shared[i] = keys[(blockIdx.x*max_ngram) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    __syncthreads();

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
    unsigned int current_btree_start = *next_level*4;
    current_ngram++;
    key = keys_shared[current_ngram];

    //Some necessary variables
    unsigned int updated_idx;
    unsigned int size;

    //Current_btree_start == 0 means that we had UNK key (vocabID 1 which wasn't found, so we should directly go to backoff
    //@TODO we can check if key[0] == 1 when we get the score too
    if (current_btree_start == 0 && key != 0) {
        goto backoff_notriecont;
    }

    while ((key != 0 && current_ngram < max_ngram - 1 && current_btree_start != 0) || 
        (get_backoff && key != 0 && current_ngram < max_ngram && current_btree_start != 0)) {
        current_ngram++;
        updated_idx = current_btree_start + 4; //Update the index for the while loop
        //@TODO consider this for shared memory as oppposed to global mem broadcast to register
        size = *(unsigned int *)&btree_trie_mem[current_btree_start]; //The size of the current node to process.

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
            if (i == 0) {
                if (key <= entries[i]) {
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i < num_entries) {
                if (key > entries[i-1] && key <= entries[i]){
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i == num_entries) {
                //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                if (key > entries[i-1]) {
                    *found_idx = i;
                }
            }
            __syncthreads();

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Calculate the address and the size of the next child
                updated_idx += *first_child_offset*4;
                if (*found_idx == 0) {
                   size = offests_incremental[0]; //0 being found_idx but a bit faster cause we hardcode it
                } else {
                    updated_idx += offests_incremental[*found_idx - 1];
                    size = offests_incremental[*found_idx] - offests_incremental[*found_idx - 1];
                }
                __syncthreads(); //Necessary @TODO why is it necessary!?!?
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
                    __syncthreads(); //Necessary if -G is omitted
                    goto backoff_part2;
                }
            } else {
                //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
                if (i < 3) {
                    //What we are doing here is reading the correct memory location for our payload. The payload is found
                    //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                    if (*is_last) {
                        payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                            + *found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                    } else {
                        payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                            + num_entries*sizeof(unsigned int) //Skip the keys
                                + *found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                    + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                    }
                }

                key = keys_shared[current_ngram]; //@TODO this might be illegal memory access
                __syncthreads();
                current_btree_start = current_btree_start + *next_level*4;

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
            if (i == 0) {
                if (key <= entries[i]) {
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i < num_entries) {
                if (key > entries[i-1] && key <= entries[i]){
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i == num_entries) {
                //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                if (key > entries[i-1]) {
                    *found_idx = i;
                }
            }
            __syncthreads();

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Calculate the address and the size of the next child
                updated_idx += *first_child_offset*4;
                if (*found_idx == 0) {
                   size = offests_incremental[0]; //0 being found_idx but a bit faster cause we hardcode it
                } else {
                    updated_idx += offests_incremental[*found_idx - 1];
                    size = offests_incremental[*found_idx] - offests_incremental[*found_idx - 1];
                }
                __syncthreads(); //@TODO why is this necessary!?!?
            } else if (!*exact_match && is_last) {
                current_ngram++; //This is necessary so that longest match logic is kept correct since in the while loop we
                goto backoff_notriecont; //Increment this before actually finding the next level
            } else {
                // We have an exact match, so we just need to add it to the payload and be done with it
                if (i == 0) {
                    //What we are doing here is reading the correct memory location for our payload. The payload is found
                    //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                    if (*is_last) {
                        accumulated_score += *(float *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                            + *found_idx*(sizeof(float))]); //Skip the previous keys' payload
                    } else {
                        accumulated_score += *(float *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                            + num_entries*sizeof(unsigned int) //Skip the keys
                                + *found_idx*(sizeof(float))]); //Skip the previous keys' payload
                    }
                }
                __syncthreads(); //Necessary if -G compilation option is omitted
            }
        }
    }

    //Write the correct result at the end
    if (i == 0) {
        results[blockIdx.x] = accumulated_score;
    }
}

void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram) {

    unsigned int max_num_children = entries_per_node + 1;

    //Time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /*
    We have to do this to provide some degree of flexibility, whilst maintaining performance
    http://stackoverflow.com/questions/32534371/cuda-most-efficient-way-to-store-constants-that-need-to-be-parsed-as-arguments?noredirect=1#comment52933276_32534371
    http://stackoverflow.com/questions/6179295/if-statement-inside-a-cuda-kernel/6179580#6179580
    http://stackoverflow.com/questions/31569401/fastest-or-most-elegant-way-of-passing-constant-arguments-to-a-cuda-kernel?rq=1
    Instantiate templates for known things:
    */
    if (entries_per_node == 3) {
        cudaEventRecord(start);
        //gpuSearchBtree<4, 3, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 5) {
        cudaEventRecord(start);
        //gpuSearchBtree<6, 5, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 7) {
        cudaEventRecord(start);
        //gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 9) {
        cudaEventRecord(start);
        //gpuSearchBtree<10, 9, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 11) {
        cudaEventRecord(start);
        //gpuSearchBtree<12, 11, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 13) {
        cudaEventRecord(start);
        //gpuSearchBtree<14, 13, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 15) {
        cudaEventRecord(start);
        //gpuSearchBtree<16, 15, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 17) {
        cudaEventRecord(start);
        //gpuSearchBtree<18, 17, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 19) {
        cudaEventRecord(start);
        //gpuSearchBtree<20, 19, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 21) {
        cudaEventRecord(start);
        //gpuSearchBtree<22, 21, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 23) {
        cudaEventRecord(start);
        //gpuSearchBtree<24, 23, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 25) {
        cudaEventRecord(start);
        //gpuSearchBtree<26, 25, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 27) {
        cudaEventRecord(start);
        //gpuSearchBtree<28, 27, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 29) {
        cudaEventRecord(start);
        //gpuSearchBtree<30, 29, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 31) {
        unsigned int shared_memory_size = (max_num_children/2 + 1 + entries_per_node + 1 + 2 + 3 + max_ngram)*sizeof(unsigned int);
        cudaEventRecord(start);
        gpuSearchBtree<<<num_ngram_queries, max_num_children, shared_memory_size>>>(btree_trie_mem, first_lvl, keys, results, entries_per_node + 1, entries_per_node, max_ngram);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 33) {
        cudaEventRecord(start);
        //gpuSearchBtree<34, 33, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 35) {
        cudaEventRecord(start);
        //gpuSearchBtree<36, 35, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 37) {
        cudaEventRecord(start);
        //gpuSearchBtree<38, 37, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 39) {
        cudaEventRecord(start);
        //gpuSearchBtree<40, 39, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 41) {
        cudaEventRecord(start);
        //gpuSearchBtree<42, 41, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 43) {
        cudaEventRecord(start);
        //gpuSearchBtree<44, 43, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 45) {
        cudaEventRecord(start);
        //gpuSearchBtree<46, 45, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 47) {
        cudaEventRecord(start);
        //gpuSearchBtree<48, 47, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 49) {
        cudaEventRecord(start);
        //gpuSearchBtree<50, 49, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 51) {
        cudaEventRecord(start);
        //gpuSearchBtree<52, 51, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 53) {
        cudaEventRecord(start);
        //gpuSearchBtree<54, 53, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 55) {
        cudaEventRecord(start);
        //gpuSearchBtree<56, 55, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 57) {
        cudaEventRecord(start);
        //gpuSearchBtree<58, 57, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 59) {
        cudaEventRecord(start);
        //gpuSearchBtree<60, 59, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 61) {
        cudaEventRecord(start);
        //gpuSearchBtree<62, 61, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 63) {
        cudaEventRecord(start);
        //gpuSearchBtree<64, 63, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 127) {
        cudaEventRecord(start);
        //gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 255) {
        cudaEventRecord(start);
        //gpuSearchBtree<256, 255, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 511) {
        cudaEventRecord(start);
        //gpuSearchBtree<512, 511, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else {
        printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
         entries_per_node, max_ngram, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID) {
    CHECK_CALL(cudaSetDevice(deviceID));
} 