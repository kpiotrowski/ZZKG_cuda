#include <vector>


//Counts newton symbol 
__device__ unsigned long long C_nk_dev(unsigned int n, unsigned int k) {
    double result = 1;
    for (unsigned int i = 1 ; i <= k; i++) {
        result *= (n-i+1);
        result /= i;
    }
    unsigned long long r = (unsigned long long) result;
    return r;
}

//Counts newton symbol 
unsigned long long C_nk(unsigned int n, unsigned int k) {
    double result = 1;
    for (unsigned int i = 1 ; i <= k; i++) {
        result *= (n-i+1);
        result /= i;
    }
    unsigned long long r = (unsigned long long) result;
    return r;
}

//P is the combinaion number. Function returns vecor of indexes
__device__ void findCombination(unsigned int p, unsigned int n, unsigned int k, int *result) {
    int i = 0;
    for (int x=0; x<k; x++) {
        unsigned long c = C_nk_dev(n-i-1, k-x-1);
        while (p>=c) {
            p-=c;
            i++;
            c = C_nk_dev(n-i-1, k-x-1);
        }
        result[x] = i;
        i++;
    }
    return;
}