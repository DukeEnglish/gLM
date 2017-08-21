# The name of this repository will not be changed until the end of this year because my dissertation is still being checked. I probably modify the code or any content anytime. If you have any question, please connect me at (ljyduke@gamil.com) without hesitation.
# giLM 
giLM is a the GPU language model based on gLM( More details about the design and implementation can be found in this [paper](http://aclweb.org/anthology/P/P16/P16-1183.pdf), published at ACL 2016.)
giLM was designed for MODLMs

## Build
```bash
git clone https://github.com/DukeEnglish/gLM.git
cd gLM
mkdir release_build
cd release_build
cmake ..
make -j4


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
path_to_binary_lm_dir : the directory of binary_lm
path_to_test_file: the batch query file (which contains all the sentence you want to query. For single sentence you should use interactive query)
