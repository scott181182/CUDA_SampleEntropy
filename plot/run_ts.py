import re
import subprocess
import sys



if __name__ == "__main__":
    ts_length = int(sys.argv[1])
    samples = int(sys.argv[2])

    print(f"{samples} samples on n={ts_length}")

    total = 0
    for i in range(samples):
        res = subprocess.run(["./target/sampen", "noisy", "gpu", str(ts_length)], capture_output=True, encoding="utf8")
        m = re.search(r"Elapsed Time = ([0-9\.]+)s", res.stdout)
        if(m is None):
            print(f"    [i] Could not find elapsed time match")
            continue
        elapsed = m.group(1)
        print(f"    {elapsed}")
        total += float(elapsed)

    mean = total / samples
    print(f"Avg Time = {mean}s")
