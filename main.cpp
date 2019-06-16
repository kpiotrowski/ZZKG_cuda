#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include "src/config.h"
#include "src/io/load.cpp"
#include "src/data.h"
#include "src/io/print.cpp"

std::vector<char> pattern;
std::vector<int> sequence;
std::vector<struct patternResult> result;

int main(int argc, char** argv){
    if(argc < 2) {
        std::cerr << "You need to provide name of the input file as the first argument";
        return 1;
    }

    try {
        loadInput(argv[1], pattern, sequence);

        printOutput(result);
    } catch (const std::exception& e ) {
        std::cerr << e.what();
        return 1;
    }
    return 0;
}
