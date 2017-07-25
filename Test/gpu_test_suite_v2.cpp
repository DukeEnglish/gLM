//#define ARPA_TESTFILEPATH is defined by cmake
#include "tests_common.hh"
#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"
#include <memory>
#include <boost/tokenizer.hpp>

 std::unique_ptr<float[]> sent2ResultsVector(std::string& sentence, LM& lm, unsigned char * btree_trie_gpu, unsigned int * first_lvl_gpu) {
    //tokenized
    boost::char_separator<char> sep(" ");
    std::vector<std::string> tokenized_sentence;
    boost::tokenizer<boost::char_separator<char> > tokens(sentence, sep);
    for (auto word : tokens) {
        tokenized_sentence.push_back(word);
    }

    //Convert to vocab IDs
    std::vector<unsigned int> vocabIDs = sent2vocabIDs(lm, tokenized_sentence, false);

	std::vector<unsigned int> allvocabIDs = allwords(lm);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Only way to get how
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);
	printf("voabIDs%d\n",vocabIDs[0]);
	printf("queriesIDs%d\n",queries[0]);
	printf("voabIDs%d\n",vocabIDs[2]);
	printf("voabIDs%d\n",vocabIDs[3]);
    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return results_cpu;
}

 std::unique_ptr<float[]> sent2ResultsVector(std::string& sentence, GPUSearcher& engine, int streamID) {
    //tokenized
    boost::char_separator<char> sep(" ");
    std::vector<std::string> tokenized_sentence;
    boost::tokenizer<boost::char_separator<char> > tokens(sentence, sep);
    for (auto word : tokens) {
        tokenized_sentence.push_back(word);
    }

    //Convert to vocab IDs
    std::vector<unsigned int> vocabIDs = sent2vocabIDs(engine.lm, tokenized_sentence, true);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, engine.lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/engine.lm.metadata.max_ngram_order; //Only way to get how
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    engine.search(gpuKeys, num_keys, results, streamID);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return results_cpu;
}

std::pair<bool, unsigned int> checkIfSame(float * expected, float * actual, unsigned int num_entries) {
    bool all_correct = true;
    unsigned int wrong_idx = 0; //Get the index of the first erroneous element
    for (unsigned int i = 0; i < num_entries; i++) {
        if (!float_compare(expected[i], actual[i])) {
            wrong_idx = i;
            all_correct = false;
            break;
        }
    }

    return std::pair<bool, unsigned int>(all_correct, wrong_idx);
}

BOOST_AUTO_TEST_SUITE(Btree)

BOOST_AUTO_TEST_CASE(micro_LM_test_small)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 7); //Use a small amount of entries per node.
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
//    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
//    BOOST_CHECK_MESSAGE(res.first, res.second);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "he is"; //Sentence with no backoff

    float expected1[3] = {-3.08719, -2.61558, -2.49612};
    //Query on the GPU
	std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 3);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

}

BOOST_AUTO_TEST_SUITE_END()
