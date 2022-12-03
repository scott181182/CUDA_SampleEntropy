#include <stdio.h>
#include <stdlib.h>
#include <string.h>



void load_series_data(const char* filename, float* out) {
    FILE* stream = fopen(filename, "r");
    char line[64];
    unsigned int i = 0;

    while(fgets(line, 64, stream)) { out[i++] = strtof(line, NULL); }

    fclose(stream);
}
