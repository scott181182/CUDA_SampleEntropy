#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cuda.h"
#include "./book.h"



#define KERNEL_SIZE 1024



__device__ float chebyshev_distance_gpu(float* A, float* B, uint32_t length) {
    float d = 0;
    for(uint32_t i = 0; i < length; i++) {
        d = max(abs(A[i] - B[i]), d);
    }
    return d;
}

__global__ void sampen_kernel(float* data, uint32_t length, uint32_t m, float r, uint32_t *AB) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i > length - m) { return; }
    if(i + m < length) {
        // Start at -1 to account for the self-match.
        int A = -1, B = -1;
        for(uint32_t j = 0; j <= length - m; j++) {
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
        for(uint32_t j = 0; j <= length - m; j++) {
            if(chebyshev_distance_gpu(&data[i], &data[j], m) < r) { B++; }
        }
        AB[2 * i] = 0;
        AB[2 * i + 1] = B;
    }
}
float sampen_gpu(float* data, uint32_t length, uint32_t m, float r) {
    if(m < 1) { return 0; }

    float *data_dev;
    HANDLE_ERROR(cudaMalloc((void**)&data_dev, sizeof(float) * length));
    HANDLE_ERROR(cudaMemcpy(data_dev, data, sizeof(float) * length, cudaMemcpyHostToDevice));

    /** Interleaved array of window-wise {A, B} pairings. */
    uint32_t AB_length = 2 * (length - m + 1);
    size_t AB_pitch = AB_length * sizeof(uint32_t);
    uint32_t *AB_dev;
    HANDLE_ERROR(cudaMalloc((void**)&AB_dev, AB_pitch));

    const uint32_t BLOCK_SIZE = (length / KERNEL_SIZE) + 1;
    sampen_kernel<<<KERNEL_SIZE, BLOCK_SIZE>>>(data_dev, length, m, r, AB_dev);

    cudaFree(data_dev);
    uint32_t *AB_arr = (uint32_t*)malloc(AB_pitch);
    HANDLE_ERROR(cudaMemcpy(AB_arr, AB_dev, AB_pitch, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(AB_dev));

    unsigned long A = 0, B = 0;
    for(uint32_t i = 0; i < AB_length; i += 2) {
        A += AB_arr[i];
        B += AB_arr[i + 1];
    }
    free(AB_arr);

    printf("A=%ld, B=%ld\n", A, B);
    printf("A/B=%16.16lf\n", (float)A / (float)B);
    return -log((float)A / (float)B);
}





inline __host__ uint2 operator+(uint2& a, uint2& b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ void operator+=(uint2& a, uint2& b) {
    a.x += b.x;
    a.y += b.y;
}

__global__ void sampen_kernel_multi(float* data_multi, uint32_t* lengths, uint32_t max_length, uint32_t m, float r, uint2 *AB_multi) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // Which time-series this thread is operating on
    int tsi = threadIdx.y + blockDim.y * blockIdx.y;

    uint32_t length = lengths[tsi];
    if(i > length - m) { return; }

    uint32_t tsoffset = tsi * max_length;
    float *data = &data_multi[tsoffset];
    uint2 *AB   = &AB_multi[tsoffset];

    // Start at -1 to account for the self-match.
    int A = -1, B = -1;
    if(i + m < length) {
        for(uint32_t j = 0; j <= length - m; j++) {
            if(chebyshev_distance_gpu(&data[i], &data[j], m) < r) {
                B++;
                if(j + m < length && abs(data[i + m] - data[j + m]) < r) {
                    A++;
                }
            }
        }
    } else {
        A = 0;
        for(uint32_t j = 0; j <= length - m; j++) {
            if(chebyshev_distance_gpu(&data[i], &data[j], m) < r) { B++; }
        }
    }
    AB[i] = make_uint2(A, B);
}

void sampen_gpu_multi(float** data, uint32_t* lengths, uint32_t n, uint32_t m, float r, float* results) {
    if(m < 1) { return; }

    uint32_t max_length = reduce_max(lengths, n);
    // printf("Max Length = %d\n", max_length);
    float *data_dev;
    HANDLE_ERROR(cudaMalloc((void**)&data_dev, sizeof(float) * max_length * n));
    // Copy asymmetric time series in one at a time.
    for(uint32_t i = 0; i < n; i++) {
        HANDLE_ERROR(cudaMemcpy(&data_dev[i * max_length], data[i], sizeof(float) * lengths[i], cudaMemcpyHostToDevice));
    }

    uint32_t *lengths_dev;
    HANDLE_ERROR(cudaMalloc((void**)&lengths_dev, sizeof(uint32_t) * n));
    HANDLE_ERROR(cudaMemcpy(lengths_dev, lengths, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

    /** Interleaved array of window-wise {A, B} pairings. */
    size_t AB_pitch = max_length * sizeof(uint2);
    uint2 *AB_dev;
    HANDLE_ERROR(cudaMalloc((void**)&AB_dev, AB_pitch * n));

    dim3 block(KERNEL_SIZE, 1);
    dim3 grid((max_length / KERNEL_SIZE) + 1, n);
    printf("Block: (%d, %d), Grid: (%d, %d)\n", block.x, block.y, grid.x, grid.y);
    sampen_kernel_multi<<<grid, block>>>(data_dev, lengths_dev, max_length, m, r, AB_dev);
    cudaFree(data_dev);
    cudaFree(lengths_dev);

    uint2 *AB_arr = (uint2*)malloc(AB_pitch * n);
    HANDLE_ERROR(cudaMemcpy(AB_arr, AB_dev, AB_pitch * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(AB_dev));

    for(uint32_t tsi = 0; tsi < n; tsi++) {
        uint32_t tsoffset = tsi * max_length;
        uint32_t ab_length = lengths[tsi] - m + 1;
        uint2 AB_sum = make_uint2(0, 0);
        for(uint32_t i = tsoffset; i < tsoffset + ab_length; i++) {
            AB_sum += AB_arr[i];
        }
        // printf("    [%d] %d / %d\n", tsi, AB_sum.x, AB_sum.y);
        results[tsi] = log((float)AB_sum.y / (float)AB_sum.x);
    }
    free(AB_arr);

    // printf("A=%ld, B=%ld\n", A, B);
    // printf("A/B=%16.16lf\n", (float)A / (float)B);
}
