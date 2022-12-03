#include <stdio.h>
#include <stdlib.h>
#include <math.h>



float chebyshev_distance(float* A, float* B, unsigned int length) {
    float d = 0;
    for(unsigned int i = 0; i < length; i++) {
        d = max(abs(A[i] - B[i]), d);
    }
    return d;
}

float sampen_cpu(float* data, unsigned int length, unsigned int m, float r) {
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
    printf("A/B=%16.16f\n", (float)A / (float)B);
    return -log((float)A / (float)B);
}
