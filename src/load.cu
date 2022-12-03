#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./load.h"



void load_series_data(const char* filename, double* out) {
    FILE* stream = fopen(filename, "r");
    char line[64];
    unsigned int i = 0;

    while(fgets(line, 64, stream)) { out[i++] = strtod(line, NULL); }

    fclose(stream);
}