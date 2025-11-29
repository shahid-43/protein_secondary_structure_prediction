import numpy as np

data = np.load("data/cb513.npy", allow_pickle=True)
print(type(data), data.shape)
print(data)