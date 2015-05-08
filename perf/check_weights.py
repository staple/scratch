import numpy as np
import pandas as pd

def read(name):
    return pd.read_csv(name, header=None)

weights = [read(x) for x in ["agd_w.csv", "agdo_w.csv", "tfocs_w.csv", "tfocso_w.csv", "tfocsf_w.csv"]]

def stats(t, s):
    return np.linalg.norm(t - s) / np.linalg.norm(t)
    return np.sqrt(np.mean(np.square(t - s)))

print stats(weights[0], weights[1])
print stats(weights[0], weights[2])
print stats(weights[0], weights[3])
print stats(weights[0], weights[4])
