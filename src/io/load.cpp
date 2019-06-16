#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string.h>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "../config.h"


std::vector<std::string> splitLine(std::string line) {
    std::vector<std::string> result;
    boost::split(result, line, boost::is_any_of(" "), boost::token_compress_off);
    return result;
}

void loadInput(char* fileName, std::vector<char> &pattern, std::vector<int> &sequence) {
    std::string line;

    std::ifstream inputFile(fileName);
    if (inputFile.is_open()){
        if ( std::getline (inputFile, line)){
            for ( auto &i : splitLine(line) ) {
                pattern.push_back(i.c_str()[0]);
            }
        }
        if ( std::getline (inputFile, line)){
            for ( auto &i : splitLine(line) ) {
                sequence.push_back(atoi(i.c_str()));
            }
        }
        inputFile.close();
    } else {
        throw std::invalid_argument("Cannot open input file");
    }

    if (DEBUG) {
        std::cout << "Input pattern: " << std::endl;
        for ( auto &i : pattern ) std::cout << i << " ";
        std::cout << std::endl << "Input sequence: " << std::endl;
        for ( auto &i : sequence ) std::cout << i << " ";
    }
}
