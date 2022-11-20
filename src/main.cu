#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"
#include "./book.h"



#define DATA_SIZE  6502



void load_series_data(const char* filename, double* out) {
    FILE* stream = fopen(filename, "r");
    char line[64];
    unsigned int i = 0;

    while(fgets(line, 64, stream)) {
        out[i++] = strtod(line, NULL);
    }

    fclose(stream);
}

int main(int argc, char const *argv[])
{
    if(argc < 2) {
        printf("Expected filename as the first parameter\n");
        return 1;
    }

    double data[DATA_SIZE];

    const char* filename = argv[1];
    printf("Reading data from '%s'...", filename);
    load_series_data(filename, data);
    printf("done!\n");

    printf("%24.24lf\n", data[0]);
    printf("...\n");
    printf("%24.24lf\n", data[DATA_SIZE - 1]);

    return 0;
}
