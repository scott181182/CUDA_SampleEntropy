#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./load.cu"
#include "./utils.cu"
#include "./sampen_cpu.cu"
#include "./sampen_gpu.cu"



#define DATA_SIZE  6502



void run_benchmark(uint32_t n) {
    // printf("Reading multi-series data...");
    float data1[DATA_SIZE];
    float data2[DATA_SIZE];
    float data3[DATA_SIZE];
    load_series_data("data/sine.csv", data1);
    load_series_data("data/rand.csv", data2);
    load_series_data("data/real.csv", data3);
    // printf("done!\n");

    float r = 0.15 * standard_deviation(data1, DATA_SIZE);
    // printf("r=%16.16lf\n", r);

    uint32_t *lengths = (uint32_t*)malloc(sizeof(uint32_t) * n);
    float **data_multi = (float**)malloc(sizeof(float*) * n + sizeof(float) * n * DATA_SIZE);
    if(data_multi == NULL) {
        printf("Unable to allocate data matrix.\n");
        return;
    }
    for(uint32_t i = 0; i < n; i++) {
        lengths[i] = DATA_SIZE;
        data_multi[i] = ((float*)(data_multi + n) + i * DATA_SIZE);

        float* data = (i % 3) == 0 ? data1 : (i % 3) == 1 ? data2 : data3;
        memcpy(data_multi[i], data, DATA_SIZE * sizeof(float));
        // printf("    [%d] %f ... %f\n", i, data_multi[i][0], data_multi[i][DATA_SIZE - 1]);
    }
    float* results = (float*)malloc(sizeof(float) * n);

    printf("Performing MULTI(n=%d) SampEn calculation on GPU...\n\n", n);
    clock_t t = clock();
    sampen_gpu_multi(data_multi, lengths, n, 2, r, results);
    t = clock() - t;
    float elapsed = ((float)t)/CLOCKS_PER_SEC;
    printf("Elapsed Time = %8.6lfs\n", elapsed);
    for(uint32_t i = 0; i < n; i++) {
        // printf("    sampen[%d] = %8.8f\n", i, results[i]);
    }

    free(data_multi);
    free(results);
}
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


    if(strncmp(argv[1], "benchmark", 9) == 0) {
        uint32_t n = strtoul(argv[2], NULL, 10);
        run_benchmark(n);
        return 0;
    }

    const char *filename = (char*)malloc(64);
    uint32_t ts_length = DATA_SIZE;

    if(strncmp(argv[1], "noisy", 5) == 0) {
        if(argc < 4) {
            printf("Expected parameter 'n' to noisy as third parameter\n");
            free((char*)filename);
            return 1;
        }

        ts_length = strtoul(argv[3], NULL, 10);
        sprintf((char*)filename, "data/noisysine_%d.csv", ts_length);
    } else {
        memcpy((char*)filename, argv[1], strnlen(argv[1], 64));
    }

    printf("Reading data from '%s'...", filename);
    float *data = (float*)malloc(sizeof(float) * ts_length);
    load_series_data(filename, data);
    printf("done!\n");

    // printf("%24.24lf\n", data[0]);
    // printf("...\n");
    // printf("%24.24lf\n", data[DATA_SIZE - 1]);

    float r = 0.15 * standard_deviation(data, ts_length);
    printf("r=%16.16lf\n", r);

    if(strncmp(argv[2], "cpu", 3) == 0) {
        printf("Performing SampEn calculation on CPU...\n\n");
        clock_t t = clock();
        float result = sampen_cpu(data, ts_length, 2, r);
        t = clock() - t;
        float elapsed = ((float)t)/CLOCKS_PER_SEC;
        printf("Sample Entropy = %16.16lf\n", result);
        printf("Elapsed Time = %8.6lfs", elapsed);
    } else if(strncmp(argv[2], "gpu", 3) == 0) {
        printf("Performing SampEn calculation on GPU...\n\n");
        clock_t t = clock();
        float result = sampen_gpu(data, ts_length, 2, r);
        t = clock() - t;
        float elapsed = ((float)t)/CLOCKS_PER_SEC;
        printf("Sample Entropy = %16.16lf\n", result);
        printf("Elapsed Time = %8.6lfs", elapsed);
    } else if(strncmp(argv[2], "mul", 3) == 0) {
        if(argc < 4) {
            printf("Expected parameter 'n' to mul as third parameter\n");
            free((char*)filename);
            return 1;
        }

        uint32_t n = strtoul(argv[3], NULL, 10);
        uint32_t *lengths = (uint32_t*)malloc(sizeof(uint32_t) * n);
        float **data_multi = (float**)malloc(sizeof(float*) * n + sizeof(float) * n * ts_length);
        if(data_multi == NULL) {
            printf("Unable to allocate data matrix.\n");
            free((char*)filename);
            return 2;
        }
        for(uint32_t i = 0; i < n; i++) {
            lengths[i] = ts_length;
            data_multi[i] = ((float*)(data_multi + n) + i * ts_length);
            memcpy(data_multi[i], data, ts_length * sizeof(float));
            // printf("    [%d] %f ... %f\n", i, data_multi[i][0], data_multi[i][ts_length - 1]);
        }
        float* results = (float*)malloc(sizeof(float) * n);

        printf("Performing MULTI(n=%d) SampEn calculation on GPU...\n\n", n);
        clock_t t = clock();
        sampen_gpu_multi(data_multi, lengths, n, 2, r, results);
        t = clock() - t;
        float elapsed = ((float)t)/CLOCKS_PER_SEC;
        printf("Elapsed Time = %8.6lfs\n", elapsed);
        for(uint32_t i = 0; i < n; i++) {
            printf("    sampen[%d] = %8.8f\n", i, results[i]);
        }

        free(data_multi);
        free(results);
    } else {
        printf("Unknown processing mode '%s', expected 'cpu' or 'gpu'\n", argv[2]);
        free((char*)filename);
        return 1;
    }
    free((char*)filename);

    return 0;
}
