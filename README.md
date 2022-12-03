# CUDA Sample Entropy

This project is a proof-of-concept for using GPU-accelerated multi-scale sample entropy.

## Project Structure

### `data`

The `data` directory holds reference data:

* `sine.csv` is a newline-delimited file holding 6,502 samples of a sine wave
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `0.059232`
* `rand.csv` is a newline-delimited file holding 6,502 samples of a uniform distribution from 0 to 2
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `2.471621`
* `real.csv` is a newline-delimited file holding 6,502 samples of real physiological time-series data
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `0.170104`
* `gen.py` generates these three series based on `ppt1t3.csv`, which containers center-of-pressure data from a force plate

### `ref`

The `ref` directory contains some reference materials/programs in order to verify the CUDA implementation

## Results

Reference R implementation (using a C binding) takes an average of 0.222 seconds on the datasets in this projects.
My naive C implementation takes about 0.50 seconds,
My naive CUDA C++ implementation (using one thread per output template window) takes 0.145 seconds.

