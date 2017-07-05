# gLM 0.4
gLM the GPU based Language Model is an ngram language model implementation that takes in an arpa file as an input, binarizes it and queries it in batch. More details about the design and implementation can be found in this [paper](http://aclweb.org/anthology/P/P16/P16-1183.pdf), published at ACL 2016.

## Build
```bash
git clone https://github.com/XapaJIaMnu/gLM.git
cd gLM
mkdir release_build
cd release_build
cmake ..
make -j4
make test #Requires CUDA for GPU testing
```

### Additional cmake build flags
- `-DBUILDTYPE=debug` builds with -O0 and -g
- `-DCOMPUTE_VER` set the compute version of the hardware. Default is 52. **IT WILL NOT PRODUCE CORRECT SCORES IF IT IS COMPILED WITH A WRONG COMPUTE VERSION!!! CHECK YOUR GPU'S COMPUTE VERSION [HERE](https://en.wikipedia.org/wiki/CUDA)**. If `make test` doesn't fail any of the GPU tests, it means your compute version is correct.
- `-DBAD_HOST` this should help building on older Ubuntu systems such as 12.04 and 14.04. Don't use it unless you have trouble building.
- `-DPYTHON_INCLUDE_DIR` defines the path to the python library such as `/usr/include/python2.7/pyconfig.h` or `/usr/include/python3.6m/pyconfig` and enables building the python components.
- `-DPYTHON_VER` is set to default to 2.7 If you want to build the python components with a different version, set it to your desired version. It would have no effect unless `-DPYTHON_INCLUDE_DIR` is set.
- `--DYAMLCPP_DIR` should be se if your yaml-cpp is in a non standard location (standard is `/usr/incude`).


## Binarize arpa files
```bash
cd path_to_glm/release_build/bin
./binarize_v2 path_to_arpa_file output_path [btree_node_size]
```
*btree_node_size* should be an odd number. Personally I found that 31 works best, but you should experiment. The number could vary with different size arpa files and different GPUs

## Batch query
To benchmark gLM in batch setting do:
```bash
cd path_to_glm/release_build/bin
./batch_query_v2 path_to_binary_lm_dir path_to_test_file [gpuDeviceID=0] [addBeginEndMarkers_bool=1] //[default setup]
```
This will calculate the perplexity of a text file. If *gpuDeviceID* is set, it will tell the gpu portion of the code to be executed on a particular GPU. You can check the available gpus on a system using the `nvidia_smi` command. 0 is a safe default to have if you want to set it. If *add_begin_end_markers* is set to 0, the begin of sentence and end of sentence tokens (\<s\> and \</s\>) will not surround every sentence.

## Preliminary results
So... Everything started running correctly. A (preliminary) benchmark against single threaded probing KenLM (Titan X vs core i7 4720HQ)

| LM  | ngram queries per second | model info          |
|-----|:------------------------:| -------------------:|
|KenLM| 10 274 237               |3.3G, 88720517 ngrams|
|gLM  | 65 459 102               |3.3G, 88720517 ngrams|


Multithreaded benchmark, same GPU against 2x Intel(R) Xeon(R) CPU E5-2680 0 @ 2.70GHz

| LM  | ngram queries per second | model info          |
|-----|:------------------------:| -------------------:|
|KenLM 1 Thread| 8 310 761               |3.3G, 88720517 ngrams|
|KenLM 2 Thread| 15 823 376               |3.3G, 88720517 ngrams|
|KenLM 4 Thread| 27 201 337               |3.3G, 88720517 ngrams|
|KenLM 8 Thread| 43 336 444               |3.3G, 88720517 ngrams|
|KenLM 16 Thread| 49 218 076               |3.3G, 88720517 ngrams|
|KenLM 32 Thread| 119 539 677               |3.3G, 88720517 ngrams|
|gLM  | 65 459 102               |3.3G, 88720517 ngrams|

Scheduling issue likely causes the low performance in 16 thread case. gLM achieves 2 times better performance relative to the cost of the hardware. ($1000 for the GPU vs $3500 for the CPUs)

## Changelog
* Version 0.1
  * Initial release.
* Version 0.2
  * Completely rewrote the BTree construction algorithm to make it faster, deterministic and also producing better, more regular BTrees.
  * First Trie level is now an array.
  * Performance has improved ~2x compared to the previous release.
* Version 0.3
  * Export example python bindings of the LM. 
* Version 0.4
  * Fix a rare issue that would cause crashes or incorrect ngrams with some small datasets.
  * Provided a proper GPUSearcher class (look at gpu/gpu_search_v2.hh) that simplifies querying the LM.
  * Allow for both probabilities and log probabilities to be returned when querying.
  * C++ fakeRNN class to be used for integration inside some neuralMT toolkits.
