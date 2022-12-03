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

void load_csv_data(const char* filename, float** out, unsigned int n) {
    FILE* stream = fopen(filename, "r");
    if(stream == NULL) {
        printf("Failed to open file '%s'", filename);
        exit(1);
    }

    unsigned int buffer_size = n * 20;
    char *buffer = (char*)malloc(buffer_size);
    char *record;
    unsigned int i = 0, j = 0;

    for(j = 0; fgets(buffer, buffer_size, stream) != NULL; j++) {
        record = strtok(buffer, ",");
        for(i = 0; record != NULL && i < n; i++) {
            out[i][j] = strtof(buffer, NULL);
            record = strtok(NULL, ",");
        }
    }

    free(buffer);
    fclose(stream);
}
