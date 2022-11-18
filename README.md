# CUDA Sample Entropy

This project is a proof-of-concept for using GPU-accelerated multi-scale sample entropy.

## Project Structure

### `data`

The `data` directory holds reference data:

* `sine.csv` is a newline-delimited file holding 6,502 samples of a sine wave
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `0.05897264243576966`
* `rand.csv` is a newline-delimited file holding 6,502 samples of a uniform distribution from 0 to 2
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `2.471216210156944`
* `real.csv` is a newline-delimited file holding 6,502 samples of real physiological time-series data
  * Expected sample entropy (`m=2`, `r=0.15 * SD`) of `0.17001913575028854`
* `gen.py` generates these three series based on `ppt1t3.csv`, which containers center-of-pressure data from a force plate

### `ref`

The `ref` directory contains some reference materials/programs in order to verify the CUDA implementation
