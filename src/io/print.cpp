#include <vector>
#include <iostream>
#include "../data.h"

void printOutput(std::vector<struct patternResult> result) {
    for (int i=0; i<result.size(); i++) {
        for (int j=0; i<result[i].result.size(); j++){
            std::cout << result[i].result[j].first << "=" << result[i].result[j].second << " ";
        }
        std::cout << ": ";
        for (int j=0; i<result[i].sequence.size(); j++){
            std::cout << result[i].sequence[j] << " ";
        }
        std::cout << std::endl;
    }
}