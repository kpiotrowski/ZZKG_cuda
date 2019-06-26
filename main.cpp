#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include "src/config.h"
#include "src/io/load.cpp"
#include "src/io/print.cpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "src/cuda/pattern.cpp"

int getPatternMask(thrust::host_vector<char> pattern, thrust::host_vector<int> *patternMask) {
    int patternLen = 0;
    patternMask->resize(pattern.size());
    thrust::fill(patternMask->begin(), patternMask->end(), 1);
    
    for (int i = 0 ; i<pattern.size(); i++){
        patternLen += (*patternMask)[i];
        for(int j=i+1 ; j<pattern.size(); j++) {
            if (pattern[i] == pattern[j]) (*patternMask)[j] = 0;
        }
    }
    return patternLen;
}

int main(int argc, char** argv){
    if(argc < 2) {
        std::cerr << "You need to provide name of the input file as the first argument";
        return 1;
    }
    try {
        thrust::host_vector<char> pattern;
        thrust::host_vector<int> patternMask;
        thrust::host_vector<int> sequence;
        thrust::device_vector<char> devPattern;
        thrust::device_vector<int> devSequence;
        thrust::device_vector<int> results;

        //Loads 
        loadInput(argv[1], pattern, sequence);

        //Copy pattern and sequence to GPU
        devPattern = pattern;
        devSequence = sequence;

        //Finds all combinations that match the pattern
        findPatternResults(devPattern, devSequence, &results);
        if (DEBUG) std::cout << "Found " << results.size() << " results" << std::endl;

        //Generate pttern mask to ommit redundant variables in the output
        int patternLen = getPatternMask(pattern, &patternMask);
        thrust::host_vector<int> output = decode_results(devPattern, devSequence, results, patternLen, patternMask);
                    
        printOutput(output, pattern, patternMask, results.size(), patternLen);
    } catch (const std::exception& e ) {
        std::cerr << e.what();
        return 1;
    }
    return 0;
}
