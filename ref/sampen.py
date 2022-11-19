import sys
import time

import numpy as np
import pandas as pd



def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0
    
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected filename of data file as first argument", file=sys.stderr)
        sys.exit(1)
    filename = sys.argv[1]

    print(f"Reading newline-delimited data from '{filename}'...")
    df = pd.read_csv(filename)
    XS = df.iloc[:,0].values
    m = 2
    r = 0.15 * XS.std()

    print(f"Read {len(XS)} values, calculating sample entropy (m={m}, r={r})...")
    start = time.time()
    xse = sampen(XS, m, r)
    elapsed = time.time() - start
    print(f"Sample Entropy: {xse}")
    print(f"Elapsed Time: {elapsed}s")
