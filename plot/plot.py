import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_1series():
    df = pd.read_csv("data_1.csv")
    LS = df["length"].values
    RS = df["r"].values
    CS = df["gpu"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(LS, RS, label="R (TSEntropies)")
    ax.plot(LS, CS, label="CUDA")

    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Time Series Length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([ 1000, 3000, 10000, 25000 ])
    ax.set_xticklabels([ "1000", "3000", "10000", "25000" ])
    ax.set_yticks([ 0.010, 0.100, 1, 10 ])
    ax.set_yticklabels([ "0.01", "0.10", "1.00", "10.00" ])

    ax.set_title("Sample Entropy Performance vs. Time Series Data Points")
    ax.grid()
    ax.legend()

    plt.show()

def plot_nseries():
    df = pd.read_csv("data_n.csv")
    NS = df["n"].values
    TS = df["time"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(NS, TS)

    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Number of Time Series")
    ax.set_xscale("log")
    # ax.set_xticks([ 1000, 3000, 10000, 25000 ])
    # ax.set_xticklabels([ "1000", "3000", "10000", "25000" ])

    ax.set_title("Sample Entropy Performance on Multiple Time Series (w/ 6,502 data points)")
    ax.grid()

    plt.show()

plot_1series()
plot_nseries()
