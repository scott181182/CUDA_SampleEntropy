#include <math.h>

#include "./utils.h"



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
