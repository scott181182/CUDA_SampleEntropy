#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "./sampen_cpu.h"



double chebyshev_distance(double* A, double* B, unsigned int length) {
    double d = 0;
    for(unsigned int i = 0; i < length; i++) {
        d = max(abs(A[i] - B[i]), d);
    }
    return d;
}

double sampen_cpu(double* data, unsigned int length, unsigned int m, double r) {
    if(m < 1) { return 0; }
    unsigned long A = 0, B = 0;

    for(unsigned int i = 0; i <= length - m; i++) {
        for(unsigned int j = 0; j <= length - m; j++) {
            if(i == j) { continue; }

            if(chebyshev_distance(&data[i], &data[j], m) < r) {
                B++;
                if(i + m < length && j + m < length && abs(data[i + m] - data[j + m]) < r) {
                    A++;
                }
            }
        }
    }

    printf("A=%ld, B=%ld\n", A, B);
    printf("A/B=%16.16lf\n", (double)A / (double)B);
    return -log((double)A / (double)B);
}
