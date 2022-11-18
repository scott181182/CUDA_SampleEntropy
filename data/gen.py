import numpy as np
import pandas as pd



def output_data(data: list[float], filename: str):
    with open(filename, "w") as file:
        file.writelines(map(lambda n: str(n) + "\n", data))

data = pd.read_csv("pp1t3.csv")

def get_sample_data() -> np.ndarray:
    return data.iloc[:,0].values

def get_sine_data():
    TS = data.iloc[:,3].values / 1000
    return np.sin(TS)

def get_random_data():
    return np.random.uniform(0, 2, data.shape[0])

output_data(get_sample_data(), "real.csv")
output_data(get_sine_data(), "sine.csv")
output_data(get_random_data(), "rand.csv")
