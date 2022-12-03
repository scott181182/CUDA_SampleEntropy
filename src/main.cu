#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./load.cu"
#include "./utils.cu"
#include "./sampen_cpu.cu"
#include "./sampen_gpu.cu"



#define DATA_SIZE  6502



int main(int argc, char const *argv[])
{
    if(argc < 2) {
        printf("Expected filename as the first parameter\n");
        return 1;
    }
    if(argc < 3) {
        printf("Expected process mode 'cpu' or 'gpu' as second parameter\n");
        return 1;
    }

    double data[DATA_SIZE];

    const char* filename = argv[1];
    printf("Reading data from '%s'...", filename);
    load_series_data(filename, data);
    printf("done!\n");

    // printf("%24.24lf\n", data[0]);
    // printf("...\n");
    // printf("%24.24lf\n", data[DATA_SIZE - 1]);

    double r = 0.15 * standard_deviation(data, DATA_SIZE);
    printf("r=%16.16lf\n", r);

    if(strncmp(argv[2], "cpu", 3) == 0) {
        printf("Performing SampEn calculation on CPU...\n\n");
        clock_t t = clock();
        double result = sampen_cpu(data, DATA_SIZE, 2, r);
        t = clock() - t;
        double elapsed = ((double)t)/CLOCKS_PER_SEC;
        printf("Sample Entropy = %16.16lf\n", result);
        printf("Elapsed Time = %8.6lfs", elapsed);
    } else if(strncmp(argv[2], "gpu", 3) == 0) {
        printf("Performing SampEn calculation on GPU...\n\n");
        clock_t t = clock();
        double result = sampen_gpu(data, DATA_SIZE, 2, r);
        t = clock() - t;
        double elapsed = ((double)t)/CLOCKS_PER_SEC;
        printf("Sample Entropy = %16.16lf\n", result);
        printf("Elapsed Time = %8.6lfs", elapsed);
    } else {
        printf("Unknown processing mode '%s', expected 'cpu' or 'gpu'\n", argv[2]);
        return 1;
    }

    return 0;
}
