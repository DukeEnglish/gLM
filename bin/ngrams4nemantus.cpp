#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

void doGPUWork(std::vector<unsigned int>& queries, LM& lm, unsigned char * btree_trie_gpu, unsigned int * first_lvl_gpu) {
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    //Search GPU
    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy results to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    //Free memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);
}

int main(int argc, char* argv[]) {
    if (!(argc != 5 || argc != 6)) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir path_to_ngrams_file path_to_vocab_file maxGPUMemoryMB [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 6) {
        setGPUDevice(atoi(argv[5]));
    }

    //Total GPU memory allowed to use (in MB):
    unsigned int gpuMemLimit = std::stoull(argv[4]);

    //Create the models
    LM lm(argv[1]);
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());
    unsigned int modelMemoryUsage = lm.metadata.byteArraySize/(1024*1024) +  (lm.metadata.intArraySize*4/(1024*1024)); //GPU memory used by the model in MB
    unsigned int queryMemory = gpuMemLimit - modelMemoryUsage; //How much memory do we have for the queries

    unsigned int unktoken = lm.encode_map.find(std::string("<unk>"))->second; //find the unk token for easy reuse

    //Read the vocab file
    std::unordered_map<unsigned int, std::string> softmax_vocab;
    readDatastructure(softmax_vocab, argv[3]);

    //This vector contains the softmax vocabulary in order in gLM vocab format.
    std::vector<unsigned int> softmax_vocab_vec;
    softmax_vocab_vec.reserve(softmax_vocab.size());
    for (unsigned int i = 0; i < softmax_vocab.size(); i++) {
        //We should be using an ordered map but I don't have a template for it. Sue me.
        std::string softmax_order_string = softmax_vocab.find(i)->second;
        auto mapiter = lm.encode_map.find(softmax_order_string);
        if (mapiter != lm.encode_map.end()) {
            softmax_vocab_vec.push_back(mapiter->second);
        } else {
            softmax_vocab_vec.push_back(unktoken);
        }
    }

    //Read in the ngrams file and convert to gLM vocabIDs
    //We need to replace their UNK with ours, replace BoS with 0s
    std::vector<std::vector<unsigned int> > orig_queries;
    std::ifstream is(argv[2]);

    if (is.fail() ){
        std::cerr << "Failed to open file " << argv[2] << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    getline(is, line);
    while (getline(is, line)) {
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
        boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

        std::vector<unsigned int> orig_query;
        orig_query.reserve(lm.metadata.max_ngram_order);
        while(it != tokens.end() ){
            std::string vocabItem = *it;
            if (vocabItem == "<s>") {
                orig_query.push_back(0);
            } else {
                auto mapiter = lm.encode_map.find(vocabItem);
                if (mapiter != lm.encode_map.end()) {
                    orig_query.push_back(mapiter->second);
                } else {
                    orig_query.push_back(unktoken);
                }
            }
            it++;
        }
        assert(orig_query.size() == lm.metadata.max_ngram_order); //Sanity check
        orig_queries.push_back(orig_query);
    }

    //Close the stream after we are done.
    is.close();

    //Now we need to expand the queries. Basically every lm.metadata.max_ngram_order word (starting from the first)
    //needs to be replaced byt the full softmax layer
    std::cout << "Total memory required: " << orig_queries.size()*softmax_vocab_vec.size()*lm.metadata.max_ngram_order*4/(1024*1024) << " MB." << std::endl;
    std::cout << "We are allowed to use " << gpuMemLimit << "MB out of which the model takes: " << modelMemoryUsage << "MB leaving us with: "
    << queryMemory << "MB to use for queries." << std::endl;
    //Actually we can use a bit less than queryMemory for our queries, because we need to allocate an array on the GPU that will hold the results, so 
    //We need to calculate that now. Results memory is 1/max_ngram_order of the queryMemory (one float for every max_ngram_order vocab IDs)
    unsigned int queries_memory = (queryMemory*lm.metadata.max_ngram_order)/(lm.metadata.max_ngram_order + 1);
    unsigned int results_memory = queryMemory - queries_memory;
    std::cout << "Query memory: " << queries_memory << "MB. Results memory: " << results_memory << "MB." << std::endl;

    std::vector<unsigned int> all_queries;
    all_queries.reserve((queries_memory*1024*1024 +4)/4);

    for(auto orig_query : orig_queries) {
        for (unsigned int softmax_vocab_id : softmax_vocab_vec) {
            assert(softmax_vocab_id != 0); //Sanity check
            all_queries.push_back(softmax_vocab_id);
            for (unsigned int i = 1; i < orig_query.size(); i++) { //The first is updated, the rest are the same
                all_queries.push_back(orig_query[i]);
            }
            assert(all_queries.size() % lm.metadata.max_ngram_order == 0); //Sanity check
        }
        if (all_queries.size()*4/(1024*1024) >= queries_memory) {
            //Flush the vector, send the queries to the GPU and write them to a file.
            doGPUWork(all_queries, lm, btree_trie_gpu, first_lvl_gpu);
            all_queries.clear(); //
            all_queries.reserve((queries_memory*1024*1024 +4)/4);
        }
    }

    //Now split it in sections and query it on the GPU

    //Free GPU memory
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    return 0;
}
