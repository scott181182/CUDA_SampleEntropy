#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "./book.h"



#define DATA_SIZE  6502
#define KERNEL_SIZE 1024
#define BLOCK_SIZE (DATA_SIZE / KERNEL_SIZE + 1)

__device__ double chebyshev_distance(double* A, double* B, unsigned int length) {
    double d = 0;
    for(unsigned int i = 0; i < length; i++) {
        d = max(abs(A[i] - B[i]), d);
    }
    return d;
}

double mean(double* data, unsigned int length) {
    double mu = 0;
    for(int i = 0; i < length; i++) {
        mu += data[i] / length;
    }
    return mu;
}
double standard_deviation(double* data, unsigned int length) {
    double mu = mean(data, length);
    double sigma = 0, v;
    for(int i = 0; i < length; i++) {
        v = data[i] - mu;
        sigma += v * v;
    }
    return sqrt(sigma / (length - 1));
}

__global__ void sampen_kernel(double* data, unsigned int length, unsigned int m, double r, double *A, double *B) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i > length - m) { return; }
    for(unsigned int j = 0; j <= length - m; j++) {
        if(i == j) { continue; }

        if(chebyshev_distance(&data[i], &data[j], m) < r) {
            B[i]++;
            if(i + m < length && j + m < length && abs(data[i + m] - data[j + m]) < r) {
                A[i]++;
            }
        }
    }
}
double sampen(double* data, unsigned int length, unsigned int m, double r) {
    if(m < 1) { return 0; }

    unsigned int A_length = length - m;
    unsigned int B_length = length - m + 1;
    double *data_dev, *A_dev, *B_dev;
    HANDLE_ERROR(cudaMalloc((void**)&data_dev, sizeof(double) * length));
    HANDLE_ERROR(cudaMalloc((void**)&A_dev,    sizeof(double) * A_length));
    HANDLE_ERROR(cudaMalloc((void**)&B_dev,    sizeof(double) * B_length));
    HANDLE_ERROR(cudaMemcpy(data_dev, data, sizeof(double) * length, cudaMemcpyHostToDevice));

    sampen_kernel<<<KERNEL_SIZE, BLOCK_SIZE>>>(data_dev, length, m, r, A_dev, B_dev);

    cudaFree(data_dev);
    double *A_arr = (double*)malloc(sizeof(double) * A_length);
    double *B_arr = (double*)malloc(sizeof(double) * B_length);
    HANDLE_ERROR(cudaMemcpy(A_arr, A_dev, sizeof(double) * A_length, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(B_arr, B_dev, sizeof(double) * B_length, cudaMemcpyDeviceToHost));
    cudaFree(A_dev);
    cudaFree(B_dev);

    unsigned long A = 0, B = 0;
    unsigned int i;
    for(i = 0; i < A_length; i++) {
        A += A_arr[i];
        B += B_arr[i];
    }
    B += B_arr[i];

    printf("A=%ld, B=%ld\n", A, B);
    printf("A/B=%16.16lf\n", (double)A / (double)B);
    free(A_arr);
    free(B_arr);
    return -log((double)A / (double)B);
}

void load_series_data(const char* filename, double* out) {
    FILE* stream = fopen(filename, "r");
    char line[64];
    unsigned int i = 0;

    while(fgets(line, 64, stream)) { out[i++] = strtod(line, NULL); }

    fclose(stream);
}

int main(int argc, char const *argv[])
{
    if(argc < 2) {
        printf("Expected filename as the first parameter\n");
        return 1;
    }

    double data[DATA_SIZE];

    const char* filename = argv[1];
    printf("Reading data from '%s'...", filename);
    load_series_data(filename, data);
    printf("done!\n");

    double r = 0.15 * standard_deviation(data, DATA_SIZE);
    printf("r=%16.16lf\n", r);

    printf("Calculating sample entropy...\n\n");
    clock_t t = clock();
    double result = sampen(data, DATA_SIZE, 2, r);
    t = clock() - t;
    double elapsed = ((double)t)/CLOCKS_PER_SEC;
    printf("Sample Entropy = %16.16lf\n", result);
    printf("Elapsed Time = %8.6lfs", elapsed);

    return 0;
}
