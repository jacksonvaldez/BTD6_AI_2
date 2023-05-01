import numpy as np

# Creates a random set of filter values
filters1 = np.random.uniform(-0.5, 0.5, (3, 3, 3, 16)) # Convolution layer 1
filters2 = np.random.uniform(-0.5, 0.5, (3, 3, 16, 32)) # Convolution layer 2

# Creates a set of biases, all 0 to start
biases1 = np.full(16, 0, dtype=np.float64).reshape(16, 1) # Convolution layer 1
biases2 = np.full(32, 0, dtype=np.float64).reshape(32, 1) # Convolution layer 2

np.save('trained_params/filters1.npy', filters1)
np.save('trained_params/filters2.npy', filters2)
np.save('trained_params/biases1.npy', biases1)
np.save('trained_params/biases2.npy', biases2)