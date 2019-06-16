#include <vector>
#include <iostream>
#include "../data.h"

void printOutput(std::vector<struct patternResult> result) {
    for (auto &i : result) {
        for( auto &j: i.result){
            std::cout << j.first << "=" << j.second << " ";
        }
        std::cout << ": ";
        for( auto &j: i.sequence){
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}