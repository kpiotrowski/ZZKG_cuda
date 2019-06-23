#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string.h>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <boost/algorithm/string/classification.hpp>
#include "../config.h"


std::vector<std::string> splitLine(std::string line) {
    std::vector<std::string> result;
    boost::split(result, line, boost::is_any_of(" "), boost::token_compress_off);
    return result;
}

void loadInput(char* fileName, thrust::host_vector<char> &pattern, thrust::host_vector<int> &sequence) {
    std::string line;

    std::ifstream inputFile(fileName);
    if (inputFile.is_open()){
        if ( std::getline (inputFile, line)){
            std::vector<std::string> sl = splitLine(line);
            for (int i = 0; i<sl.size(); i++) {
                pattern.push_back(sl[i].c_str()[0]);
            }
        }
        if ( std::getline (inputFile, line)){
            std::vector<std::string> sl = splitLine(line);
            for (int i = 0; i<sl.size(); i++) {
                sequence.push_back(atoi(sl[i].c_str()));
            }
        }
        inputFile.close();
    } else {
        throw std::invalid_argument("Cannot open input file");
    }

    if (DEBUG) {
        std::cout << "Input pattern: " << std::endl;
        for (int i=0; i<pattern.size(); i++)
            std::cout << pattern[i] << " ";
        std::cout << std::endl << "Input sequence: " << std::endl;
        for (int i=0; i<sequence.size(); i++)
            std::cout << sequence[i] << " ";
    }
}
