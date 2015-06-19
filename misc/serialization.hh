#include <iostream>
#include <fstream>

//Use boost to serialize the vectors
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

void SerializeByteArray(std::vector<unsigned char>& byte_arr, const char * path){
    std::ofstream os (path, std::ios::binary);
    boost::archive::text_oarchive oarch(os);
    oarch << byte_arr;
    os.close();
}

//The byte_arr vector should be reserved to prevent constant realocation.
void ReadByteArray(std::vector<unsigned char>& byte_arr, const char * path){
    std::ifstream is (path, std::ios::binary);
    boost::archive::text_iarchive iarch(is);
    iarch >> byte_arr;
    is.close();
}
