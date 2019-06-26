#include <vector>
#include <iostream>

void printOutput(
    thrust::host_vector<int> output,
    thrust::host_vector<char> pattern,
    thrust::host_vector<int> patternMask,
    int resultsLen, int patternLen
    ) {
    for (int i=0; i<resultsLen; i++){
        int output_idx = i*(patternLen+2);

        for (int j=0; j<patternMask.size(); j++) {
            if (patternMask[j] == 1) {
                std::cout << pattern[j] << "=" << output[output_idx] << " ";
                output_idx++;
            }
        }
        std::cout << ": [" << output[output_idx] << ":" << output[output_idx+1] << "]" << std::endl;
    }
}