#ifndef __H_SAMPEN_GPU__
#define __H_SAMPEN_GPU__

__device__ double chebyshev_distance_gpu(double* A, double* B, unsigned int length);
__global__ void sampen_kernel(double* data, unsigned int length, unsigned int m, double r, unsigned int *AB);
double sampen_gpu(double* data, unsigned int length, unsigned int m, double r);

#endif
