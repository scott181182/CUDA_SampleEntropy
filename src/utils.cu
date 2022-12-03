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
