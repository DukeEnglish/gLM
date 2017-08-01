//#define ARPA_TESTFILEPATH is defined by cmake
#include "tests_common.hh"
#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"
#include <memory>
#include <boost/tokenizer.hpp>
#include <stdio.h>

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
	//Convert allwords into vocab IDs
	std::vector<unsigned int> allvocabIDs = allwords(lm);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Only way to get how
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
	unsigned int results_size = allvocabIDs.size()+1;
    allocateGPUMem(results_size, &results);
	cudaMemset(results, 0, results_size*sizeof(float));
    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[results_size]);
    copyToHostMemory(results, results_cpu.get(), results_size);
	
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

	std::unique_ptr<float[]> results_test(new float[80]);
	int test=0;
	for (unsigned int i =1;i<results_size;i++){
		
		if (results_cpu[i]!=0){
			printf("results[%d]: %f\n",i,results_cpu[i]);
			results_test[test] = results_cpu[i];
			test++;
		}
	}	
	if (test == 0){
		results_test[0]=0;
	}
//    return results_cpu;
	return results_test;
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
	//Convert allwords into vocab IDs
	std::vector<unsigned int> allvocabIDs = allwords(lm);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, engine.lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/engine.lm.metadata.max_ngram_order; //Only way to get how
	 
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
	float * results;
	unsigned int results_size = allvocabIDs.size()+1;
    allocateGPUMem(results_size, &results);
	cudaMemset(results, 0, results_size*sizeof(float));

    engine.search(gpuKeys, num_keys, results, streamID);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    std::unique_ptr<float[]> results_test(new float[80]);
	int test=0;
	for (unsigned int i =1;i<results_size;i++){
		
		if (results_cpu[i]!=0){
			printf("results[%d]: %f\n",i,results_cpu[i]);
			results_test[test] = results_cpu[i];
			test++;
		}
	}	
	if (test == 0){
		results_test[0]=0;
	}
//    return results_cpu;
	return results_test;
}
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

    std::string sentence1 = "<s> he has just";//sentence which is (N-1)gram
	std::string sentence2 = "<s>";//sentence which start with a word stored in first_lvl, it need to be seek in second Btree
	std::string sentence3 = "<s> he";//sentence wihch start with a word store in second trie_level
	std::string sentence4 = "<s> he </s>";//unk word follow with a word stroed in second trie_level
	std::string sentence5 = "<s> </s>";//only a word store in first_lvl before a unk word
	std::string sentence6 = "</s> he"; //start from unkword
	std::string sentence7 = "he"; //a word store in first_lvl and doesn't need return second Btree_level

    float expected1[1] = {-0.78854};
	float expected2[78] = {-3.17068, -2.07624, -1.02523, -0.90749, -2.88672, -2.10771, -1.92893, -1.35595, -1.54547, -2.62890, -1.58193, -3.08719, -1.34876, -2.53122, -1.82026, -3.08719, -2.63453, -3.00963, -1.66855, -1.09187, -1.73340, -2.92395, -2.88672, -1.66000, -3.00963, -1.47258, -2.83625, -2.62891, -1.95528, -2.96468, -1.73466, -2.55423, -2.57247, -2.61658, -1.14549, -2.13967, -2.98657, -2.73725, -2.13967, -2.54726, -2.62776, -3.08719, -2.92395, -2.59924, -1.74742, -2.62776, -2.62891, -1.95737, -2.88672, -1.95112, -3.08719, -2.77695, -3.08719, -2.85243, -2.98657, -3.03397, -3.14785, -3.15165, -3.14785, -3.17068, -3.15165, -3.11646, -3.17068, -3.11646, -3.08719, -3.15165, -3.147848, -2.54726, -3.14785, -3.15165, -3.17068, -3.17068, -3.17068, -3.17068, -2.63453, -3.17068, -3.17068};
	float expected3[3] = {-0.95153};
	float expected4[1] = {0};//just return 0 for unk word
	float expected5[1] = {0};
	float expected6[1] = {0};
	float expected7[3] = {-1.42482, -0.38527, -1.44945};


    //Query on the GPU
	std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_5 = sent2ResultsVector(sentence5, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_6 = sent2ResultsVector(sentence6, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_7 = sent2ResultsVector(sentence7, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);
    is_correct = checkIfSame(expected2, res_2.get(), 78);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

	is_correct = checkIfSame(expected5, res_5.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 5: Expected: "
        << expected5[is_correct.second] << ", got: " << res_5[is_correct.second]);

	is_correct = checkIfSame(expected6, res_6.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 6: Expected: "
        << expected6[is_correct.second] << ", got: " << res_6[is_correct.second]);

	is_correct = checkIfSame(expected7, res_7.get(), 3);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 7: Expected: "
        << expected7[is_correct.second] << ", got: " << res_7[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

}

BOOST_AUTO_TEST_CASE(micro_LM_test_large)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 127); //Use a small amount of entries per node.
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
//    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
//    BOOST_CHECK_MESSAGE(res.first, res.second);

    std::string sentence1 = "<s> he has just";//sentence which is (N-1)gram
	std::string sentence2 = "<s>";//sentence which start with a word stored in first_lvl, it need to be seek in second Btree
	std::string sentence3 = "<s> he";//sentence wihch start with a word store in second trie_level
	std::string sentence4 = "<s> he </s>";//unk word follow with a word stroed in second trie_level
	std::string sentence5 = "<s> </s>";//only a word store in first_lvl before a unk word
	std::string sentence6 = "</s> he"; //start from unkword
	std::string sentence7 = "he"; //a word store in first_lvl and doesn't need return second Btree_level

    float expected1[1] = {-0.78854};
	float expected2[78] = {-3.17068, -2.07624, -1.02523, -0.90749, -2.88672, -2.10771, -1.92893, -1.35595, -1.54547, -2.62890, -1.58193, -3.08719, -1.34876, -2.53122, -1.82026, -3.08719, -2.63453, -3.00963, -1.66855, -1.09187, -1.73340, -2.92395, -2.88672, -1.66000, -3.00963, -1.47258, -2.83625, -2.62891, -1.95528, -2.96468, -1.73466, -2.55423, -2.57247, -2.61658, -1.14549, -2.13967, -2.98657, -2.73725, -2.13967, -2.54726, -2.62776, -3.08719, -2.92395, -2.59924, -1.74742, -2.62776, -2.62891, -1.95737, -2.88672, -1.95112, -3.08719, -2.77695, -3.08719, -2.85243, -2.98657, -3.03397, -3.14785, -3.15165, -3.14785, -3.17068, -3.15165, -3.11646, -3.17068, -3.11646, -3.08719, -3.15165, -3.147848, -2.54726, -3.14785, -3.15165, -3.17068, -3.17068, -3.17068, -3.17068, -2.63453, -3.17068, -3.17068};
	float expected3[3] = {-0.95153};
	float expected4[1] = {0};//just return 0 for unk word
	float expected5[1] = {0};
	float expected6[1] = {0};
	float expected7[3] = {-1.42482, -0.38527, -1.44945};


    //Query on the GPU
	std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_5 = sent2ResultsVector(sentence5, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_6 = sent2ResultsVector(sentence6, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_7 = sent2ResultsVector(sentence7, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);
    is_correct = checkIfSame(expected2, res_2.get(), 78);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

	is_correct = checkIfSame(expected5, res_5.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 5: Expected: "
        << expected5[is_correct.second] << ", got: " << res_5[is_correct.second]);

	is_correct = checkIfSame(expected6, res_6.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 6: Expected: "
        << expected6[is_correct.second] << ", got: " << res_6[is_correct.second]);

	is_correct = checkIfSame(expected7, res_7.get(), 3);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 7: Expected: "
        << expected7[is_correct.second] << ", got: " << res_7[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

}

BOOST_AUTO_TEST_CASE(micro_LM_test_serialization)  {
    LM lm_orig;
    createTrie(ARPA_TESTFILEPATH, lm_orig, 7); //Use a small amount of entries per node.

    std::string filepath = "/tmp/";
    const long double sysTime = time(0);
    std::stringstream s;
    s << filepath << sysTime; //Use random tmp directory

    lm_orig.writeBinary(s.str());
    LM lm(s.str());

    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
//    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
//    BOOST_CHECK_MESSAGE(res.first, res.second);

    std::string sentence1 = "<s> he has just";//sentence which is (N-1)gram
	std::string sentence2 = "<s>";//sentence which start with a word stored in first_lvl, it need to be seek in second Btree
	std::string sentence3 = "<s> he";//sentence wihch start with a word store in second trie_level
	std::string sentence4 = "<s> he </s>";//unk word follow with a word stroed in second trie_level
	std::string sentence5 = "<s> </s>";//only a word store in first_lvl before a unk word
	std::string sentence6 = "</s> he"; //start from unkword
	std::string sentence7 = "he"; //a word store in first_lvl and doesn't need return second Btree_level

    float expected1[1] = {-0.78854};
	float expected2[78] = {-3.17068, -2.07624, -1.02523, -0.90749, -2.88672, -2.10771, -1.92893, -1.35595, -1.54547, -2.62890, -1.58193, -3.08719, -1.34876, -2.53122, -1.82026, -3.08719, -2.63453, -3.00963, -1.66855, -1.09187, -1.73340, -2.92395, -2.88672, -1.66000, -3.00963, -1.47258, -2.83625, -2.62891, -1.95528, -2.96468, -1.73466, -2.55423, -2.57247, -2.61658, -1.14549, -2.13967, -2.98657, -2.73725, -2.13967, -2.54726, -2.62776, -3.08719, -2.92395, -2.59924, -1.74742, -2.62776, -2.62891, -1.95737, -2.88672, -1.95112, -3.08719, -2.77695, -3.08719, -2.85243, -2.98657, -3.03397, -3.14785, -3.15165, -3.14785, -3.17068, -3.15165, -3.11646, -3.17068, -3.11646, -3.08719, -3.15165, -3.147848, -2.54726, -3.14785, -3.15165, -3.17068, -3.17068, -3.17068, -3.17068, -2.63453, -3.17068, -3.17068};
	float expected3[3] = {-0.95153};
	float expected4[1] = {0};//just return 0 for unk word
	float expected5[1] = {0};
	float expected6[1] = {0};
	float expected7[3] = {-1.42482, -0.38527, -1.44945};


    //Query on the GPU
	std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_5 = sent2ResultsVector(sentence5, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_6 = sent2ResultsVector(sentence6, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_7 = sent2ResultsVector(sentence7, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);
    is_correct = checkIfSame(expected2, res_2.get(), 78);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

	is_correct = checkIfSame(expected5, res_5.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 5: Expected: "
        << expected5[is_correct.second] << ", got: " << res_5[is_correct.second]);

	is_correct = checkIfSame(expected6, res_6.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 6: Expected: "
        << expected6[is_correct.second] << ", got: " << res_6[is_correct.second]);

	is_correct = checkIfSame(expected7, res_7.get(), 3);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 7: Expected: "
        << expected7[is_correct.second] << ", got: " << res_7[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

}

BOOST_AUTO_TEST_CASE(micro_LM_test_normal_stream)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 31); //Use a large amount of entries per node
    
    GPUSearcher engine(2, lm);

    //Test whether we can find every single ngram that we stored
//    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
//    BOOST_CHECK_MESSAGE(res.first, res.second);

    std::string sentence1 = "<s> he has just";//sentence which is (N-1)gram
	std::string sentence2 = "<s>";//sentence which start with a word stored in first_lvl, it need to be seek in second Btree
	std::string sentence3 = "<s> he";//sentence wihch start with a word store in second trie_level
	std::string sentence4 = "<s> he </s>";//unk word follow with a word stroed in second trie_level
	std::string sentence5 = "<s> </s>";//only a word store in first_lvl before a unk word
	std::string sentence6 = "</s> he"; //start from unkword
	std::string sentence7 = "he"; //a word store in first_lvl and doesn't need return second Btree_level

    float expected1[1] = {-0.78854};
	float expected2[78] = {-3.17068, -2.07624, -1.02523, -0.90749, -2.88672, -2.10771, -1.92893, -1.35595, -1.54547, -2.62890, -1.58193, -3.08719, -1.34876, -2.53122, -1.82026, -3.08719, -2.63453, -3.00963, -1.66855, -1.09187, -1.73340, -2.92395, -2.88672, -1.66000, -3.00963, -1.47258, -2.83625, -2.62891, -1.95528, -2.96468, -1.73466, -2.55423, -2.57247, -2.61658, -1.14549, -2.13967, -2.98657, -2.73725, -2.13967, -2.54726, -2.62776, -3.08719, -2.92395, -2.59924, -1.74742, -2.62776, -2.62891, -1.95737, -2.88672, -1.95112, -3.08719, -2.77695, -3.08719, -2.85243, -2.98657, -3.03397, -3.14785, -3.15165, -3.14785, -3.17068, -3.15165, -3.11646, -3.17068, -3.11646, -3.08719, -3.15165, -3.147848, -2.54726, -3.14785, -3.15165, -3.17068, -3.17068, -3.17068, -3.17068, -2.63453, -3.17068, -3.17068};
	float expected3[3] = {-0.95153};
	float expected4[1] = {0};//just return 0 for unk word
	float expected5[1] = {0};
	float expected6[1] = {0};
	float expected7[3] = {-1.42482, -0.38527, -1.44945};


    //Query on the GPU
	std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_5 = sent2ResultsVector(sentence5, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_6 = sent2ResultsVector(sentence6, lm, btree_trie_gpu, first_lvl_gpu);
	std::unique_ptr<float[]> res_7 = sent2ResultsVector(sentence7, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);
    is_correct = checkIfSame(expected2, res_2.get(), 78);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

	is_correct = checkIfSame(expected5, res_5.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 5: Expected: "
        << expected5[is_correct.second] << ", got: " << res_5[is_correct.second]);

	is_correct = checkIfSame(expected6, res_6.get(), 1);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 6: Expected: "
        << expected6[is_correct.second] << ", got: " << res_6[is_correct.second]);

	is_correct = checkIfSame(expected7, res_7.get(), 3);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 7: Expected: "
        << expected7[is_correct.second] << ", got: " << res_7[is_correct.second]);

}

BOOST_AUTO_TEST_SUITE_END()
