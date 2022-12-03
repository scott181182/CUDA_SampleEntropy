#include <math.h>



float mean(float* data, unsigned int length) {
    float mu = 0;
    for(int i = 0; i < length; i++) {
        mu += data[i] / length;
    }
    return mu;
}
float standard_deviation(float* data, unsigned int length) {
    float mu = mean(data, length);
    float sigma = 0, v;
    for(int i = 0; i < length; i++) {
        v = data[i] - mu;
        sigma += v * v;
    }
    return sqrt(sigma / (length - 1));
}

unsigned int reduce_max(unsigned int *lengths, unsigned int n) {
    unsigned int res = lengths[0];
    for(unsigned int i = 1; i < n; i++) {
        if(lengths[i] > res) { res = lengths[i]; }
    }
    return res;
}
