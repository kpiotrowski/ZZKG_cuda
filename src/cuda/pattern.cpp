#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "../math/math.cpp"
#include "../config.h"
#include <cmath>
#include <iostream>

thrust::device_vector<char> devPattern;
thrust::device_vector<int> devSequence;
thrust::device_vector<int> tempResults;
unsigned long long maxResultsLen;

// To check if it works as expected
__device__ unsigned long long threadIndex() {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z * blockDim.z + threadIdx.z;
    return bx + (by*gridDim.x) + (bz*gridDim.x*gridDim.y);
}

__global__ void pattern_kernel(
    char* p_ptr, int pSize,
    int * s_ptr, int sSize,
    int * r_ptr, unsigned long long maxIndex)
{
    unsigned long long index = threadIndex();
    if (index >= maxIndex) return;

    int *combination = new int[pSize];
    findCombination(index, sSize, pSize, combination);

    for (int i = 0 ; i<pSize-1; i++) {
        for (int j = i+1; j<pSize; j++) {
            if (p_ptr[i] == p_ptr[j] && s_ptr[combination[i]] != s_ptr[combination[j]]) {
                r_ptr[index] = 0;
                delete[] combination;
                return;  
            }
        }
    }
    r_ptr[index] = 1;
    delete[] combination;
}

__global__ void inclusive_scan_kernel (
    int *tab_ptr, unsigned long long tabLen, int offset, int iteration
) {
    unsigned long long index = threadIndex();
    if (index + offset >= tabLen) return;

    if (iteration % 2 == 1) {
        tab_ptr[index+offset+tabLen] = tab_ptr[index]+ tab_ptr[index + offset];
    } else {
        tab_ptr[index + offset] = tab_ptr[index+tabLen]+ tab_ptr[index + offset+tabLen];
    }
}

__global__ void compact_kernel (
    int *tab_ptr, unsigned long long tabLen, int *tab_result_ptr
) {
    unsigned long long index = threadIndex();
    if (index == 0) {
        if (tab_ptr[0] == 1) tab_result_ptr[0] = index;

    } else if (index < tabLen) {
        if (tab_ptr[index-1] < tab_ptr[index]) tab_result_ptr[tab_ptr[index]-1] = index;
    }
}

std::pair<dim3, dim3> calculateGrid(unsigned long long requiredThreads) {
    double requiredBlocks = ceil((double)requiredThreads/(double)BLOCK_THREADS);

    dim3 dimBlock(BLOCK_THREADS);
    dim3 dimGrid(1, 1, 1);
    if (requiredThreads < BLOCK_THREADS) {
        dimBlock.x = requiredThreads;
        return std::make_pair(dimGrid, dimBlock);
    }

    if (requiredBlocks > GRID_MAX_SIZE) {
        dimGrid.x = GRID_MAX_SIZE;
    } else dimGrid.x = requiredBlocks;
    requiredBlocks = ceil(requiredBlocks/GRID_MAX_SIZE); 
    if (requiredBlocks > GRID_MAX_SIZE) {
        dimGrid.y = GRID_MAX_SIZE;
    } else dimGrid.y = requiredBlocks;
    requiredBlocks = ceil(requiredBlocks/GRID_MAX_SIZE);
    dimGrid.z = requiredBlocks;

    return std::make_pair(dimGrid, dimBlock);
}

void findPatternResults(
    thrust::device_vector<char> devPattern, thrust::device_vector<int> devSequence,
    thrust::device_vector<int> *result
) {
    thrust::device_vector<int> tempResults;
    unsigned long long maxResultsLen;

    int patternLen = devPattern.size(), sequenceLen = devSequence.size();
    if (patternLen > sequenceLen) throw std::invalid_argument("Pattern cannot be longer than the sequence");
    if (DEBUG) std::cout << std::endl << "Pattern len: " << patternLen << " Sequence len: " << sequenceLen << std::endl;

    maxResultsLen = C_nk(sequenceLen, patternLen);
    if (maxResultsLen==0) throw std::invalid_argument("Input payload is to big");
    if (DEBUG) std::cout << std::endl << "There is " << maxResultsLen << " combinations" << std::endl;

    std::pair<dim3, dim3> grid = calculateGrid(maxResultsLen);

    if (DEBUG) std::cout << "Running pattern kernel with GRID: " << grid.first.x << ", " << grid.first.y << ", " << grid.first.z << "  Block threads: " << grid.second.x << std::endl;
    tempResults.resize((int)maxResultsLen);

    pattern_kernel<<<grid.first, grid.second >>> (
            devPattern.data().get(), devPattern.size(),
            devSequence.data().get(), devSequence.size(),
            tempResults.data().get(), maxResultsLen
    );
    thrust::inclusive_scan(tempResults.begin(), tempResults.end(), tempResults.begin());

    result->resize(tempResults[maxResultsLen-1]);
    compact_kernel<<<grid.first, grid.second >>> (
            tempResults.data().get(), maxResultsLen,
            result->data().get()
    );
}

__global__ void decode_kernel (
    int *results_ptr, int resultsLen, int pSize, int patternUniqLen, int *pattern_mask_ptr,
    int *seq_ptr, int sSize, int *output_ptr
) {
    unsigned long long index = threadIndex();
    if (index >= resultsLen) return;
    int outputIdx = (patternUniqLen+2)*index;

    int *combination = new int[pSize];
    findCombination(results_ptr[index], sSize, pSize, combination);

    int idx=0;
    for (int i = 0; i<pSize ; i++) {
        if (pattern_mask_ptr[i] == 1) {
            output_ptr[outputIdx + idx] = seq_ptr[combination[i]];
            idx++;
        }
        output_ptr[outputIdx+patternUniqLen] = combination[0];
        output_ptr[outputIdx+patternUniqLen+1] = combination[pSize-1];
    }
}


thrust::host_vector<int> decode_results(
    thrust::device_vector<char> devPattern, thrust::device_vector<int> devSequence,
    thrust::device_vector<int> results, int patternLen, thrust::host_vector<int> patternMask
) {
    int resultsLen = results.size();
    thrust::device_vector<int> devPatternMask = patternMask;
    thrust::device_vector<int> output;
    output.resize((patternLen+2) * resultsLen);

    std::pair<dim3, dim3> grid = calculateGrid(resultsLen);
    if (DEBUG) std::cout << "Running decode kernel with GRID: " << grid.first.x << ", " << grid.first.y << ", " << grid.first.z << "  Block threads: " << grid.second.x << std::endl;

    decode_kernel<<<grid.first, grid.second >>> (
            results.data().get(), resultsLen, devPattern.size(), patternLen, devPatternMask.data().get(),
            devSequence.data().get(), devSequence.size(),
            output.data().get()
    );

    thrust::host_vector<int> outputHost = output;
    return outputHost;
}