#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "./book.h"



#define KERNEL_SIZE 1024



__device__ float chebyshev_distance_gpu(float* A, float* B, unsigned int length) {
    float d = 0;
    for(unsigned int i = 0; i < length; i++) {
        d = max(abs(A[i] - B[i]), d);
    }
    return d;
}

__global__ void sampen_kernel(float* data, unsigned int length, unsigned int m, float r, unsigned int *AB) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i > length - m) { return; }
    if(i + m < length) {
        // Start at -1 to account for the self-match.
        int A = -1, B = -1;
        for(unsigned int j = 0; j <= length - m; j++) {
            if(chebyshev_distance_gpu(&data[i], &data[j], m) < r) {
                B++;
                if(j + m < length && abs(data[i + m] - data[j + m]) < r) {
                    A++;
                }
            }
        }
        AB[2 * i] = A;
        AB[2 * i + 1] = B;
    } else {
        int B = -1;
        for(unsigned int j = 0; j <= length - m; j++) {
            if(chebyshev_distance_gpu(&data[i], &data[j], m) < r) { B++; }
        }
        AB[2 * i] = 0;
        AB[2 * i + 1] = B;
    }
}
float sampen_gpu(float* data, unsigned int length, unsigned int m, float r) {
    if(m < 1) { return 0; }

    float *data_dev;
    HANDLE_ERROR(cudaMalloc((void**)&data_dev, sizeof(float) * length));
    HANDLE_ERROR(cudaMemcpy(data_dev, data, sizeof(float) * length, cudaMemcpyHostToDevice));

    /** Interleaved array of window-wise {A, B} pairings. */
    unsigned int AB_length = 2 * (length - m + 1);
    size_t AB_pitch = AB_length * sizeof(unsigned int);
    unsigned int *AB_dev;
    HANDLE_ERROR(cudaMalloc((void**)&AB_dev, AB_pitch));

    const unsigned int BLOCK_SIZE = (length / KERNEL_SIZE) + 1;
    sampen_kernel<<<KERNEL_SIZE, BLOCK_SIZE>>>(data_dev, length, m, r, AB_dev);

    cudaFree(data_dev);
    unsigned int *AB_arr = (unsigned int*)malloc(AB_pitch);
    HANDLE_ERROR(cudaMemcpy(AB_arr, AB_dev, AB_pitch, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(AB_dev));

    unsigned long A = 0, B = 0;
    for(unsigned int i = 0; i < AB_length; i += 2) {
        A += AB_arr[i];
        B += AB_arr[i + 1];
    }
    free(AB_arr);

    printf("A=%ld, B=%ld\n", A, B);
    printf("A/B=%16.16lf\n", (float)A / (float)B);
    return -log((float)A / (float)B);
}
