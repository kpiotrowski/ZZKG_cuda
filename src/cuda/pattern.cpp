#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../math/math.cpp"
#include "../config.h"
#include <cmath>
#include <iostream>


thrust::device_vector<char> devPattern;
thrust::device_vector<int> devSequence;
thrust::device_vector<int> tempResults;
unsigned long long maxResultsLen;

__global__ void pattern_kernel(
    char* p_ptr, int pSize,
    int * s_ptr, int sSize,
    int * r_ptr, unsigned long long maxIndex,
    int kernelNum,
    unsigned long long gridMaxSize)
{
    unsigned long long index = kernelNum*gridMaxSize*gridMaxSize*gridMaxSize;
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z * blockDim.z + threadIdx.z;
    index += (bx + (by*gridMaxSize) + (bz*gridMaxSize*gridMaxSize));
    
    int *combination = new int[pSize];
    findCombination(index, sSize, pSize, combination);
    if (index >= maxIndex) return;

    for (int i = 0 ; i<pSize-1; i++) {
        for (int j = i+1; i<pSize; j++) {
            if (p_ptr[i] == p_ptr[j] && s_ptr[combination[i]] != s_ptr[combination[j]]) {
                r_ptr[index] = 0;
                return;  
            }
        }
    }
    r_ptr[index] = 1;
    delete[] combination;
}


void findPatternResults(
    thrust::host_vector<char> pattern, thrust::host_vector<int> sequence
) {
    int patternLen = pattern.size(), sequenceLen = sequence.size();
    if (patternLen > sequenceLen) throw std::invalid_argument("Pattern cannot be longer than the sequence");
    if (DEBUG) std::cout << std::endl << "Pattern len: " << patternLen << " Sequence len: " << sequenceLen << std::endl;

    maxResultsLen = C_nk(sequenceLen, patternLen);
    if (maxResultsLen==0) throw std::invalid_argument("Input payload is to big");
    if (DEBUG) std::cout << std::endl << "There is " << maxResultsLen << " combinations" << std::endl;

    // devPattern = pattern;
    // devSequence = sequence;
    // tempResults.resize((int)maxResultsLen);

    double requiredBlocks = ceil((double)maxResultsLen/(double)BLOCK_THREADS);
    int kernelsNum = (int)ceil(requiredBlocks/GRID_MAX_SIZE/GRID_MAX_SIZE/GRID_MAX_SIZE);
    if (DEBUG) std::cout << "Needs " << kernelsNum << " kernels to find combination patterns" << std::endl;

    for (int i=1; i<= kernelsNum; i++) {
        dim3 dimBlock(BLOCK_THREADS);
        dim3 dimGrid(GRID_MAX_SIZE, GRID_MAX_SIZE, GRID_MAX_SIZE);
        if (i==kernelsNum) {
            double blocksToRun = requiredBlocks - (GRID_MAX_BLOCKS*(i-1));
            if (blocksToRun > GRID_MAX_SIZE) {
                dimGrid.x = GRID_MAX_SIZE;
            } else dimGrid.x = blocksToRun;
            blocksToRun = ceil(blocksToRun/GRID_MAX_SIZE); 
            if (blocksToRun > GRID_MAX_SIZE) {
                dimGrid.y = GRID_MAX_SIZE;
            } else dimGrid.y = blocksToRun;
            blocksToRun = ceil(blocksToRun/GRID_MAX_SIZE);
            dimGrid.z = blocksToRun;
        }
        if (DEBUG) std::cout << "Running kernel with GRID: " << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << "  Block threads: " << dimBlock.x << std::endl;

        // pattern_kernel<<<dimGrid, dimBlock >>> (
        //     devPattern.data().get(), devPattern.size(),
        //     devSequence.data().get(), devSequence.size(),
        //     tempResults.data().get(), maxResultsLen, 
        //     i-1, GRID_MAX_SIZE
        // );
    }

}