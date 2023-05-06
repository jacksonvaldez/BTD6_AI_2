import numpy as np

# Creates a random set of filter values
filters1 = np.random.uniform(-1.0, 1.0, (3, 3, 3, 16)) # Convolution layer 1
filters2 = np.random.uniform(-1.0, 1.0, (3, 3, 16, 32)) # Convolution layer 2

# Creates a set of biases, all 0 to start
biases1 = np.full(16, 0, dtype=np.float64).reshape(16, 1) # Convolution layer 1
biases2 = np.full(32, 0, dtype=np.float64).reshape(32, 1) # Convolution layer 2

# Creates weights for the feed forward neural network
weights1 = np.random.uniform(-1.0, 1.0, (211200, 30))
weights2 = np.random.uniform(-1.0, 1.0, (30, 4))
weights3_p1 = np.random.uniform(-1.0, 1.0, (4, 109350))
weights3_p2 = np.random.uniform(-1.0, 1.0, (4, 109350))
weights3_p3 = np.random.uniform(-1.0, 1.0, (4, 109350))
weights4_p1 = np.random.uniform(-1.0, 1.0, (109350, 23))
weights4_p2 = np.random.uniform(-1.0, 1.0, (109350, 3))

# Creates biases for the feed forward neural network
biases3 = np.random.uniform(-1.0, 1.0, (30, 1)) # Feed Forward layer 1
biases4 = np.random.uniform(-1.0, 1.0, (4, 1)) # Feed Forward layer 2
biases5_p1 = np.random.uniform(-1.0, 1.0, (109350, 1)) # Feed Forward layer 3.1
biases5_p2 = np.random.uniform(-1.0, 1.0, (109350, 1)) # Feed Forward layer 3.2
biases5_p3 = np.random.uniform(-1.0, 1.0, (109350, 1)) # Feed Forward layer 3.3
biases6_p1 = np.random.uniform(-1.0, 1.0, (23, 1)) # Feed Forward layer 4.1
biases6_p2 = np.random.uniform(-1.0, 1.0, (3, 1)) # Feed Forward layer 4.2

np.save('trained_params/filters1.npy', filters1)
np.save('trained_params/filters2.npy', filters2)
np.save('trained_params/biases1.npy', biases1)
np.save('trained_params/biases2.npy', biases2)

np.save('trained_params/weights1.npy', weights1)
np.save('trained_params/weights2.npy', weights2)
np.save('trained_params/weights3_p1.npy', weights3_p1)
np.save('trained_params/weights3_p2.npy', weights3_p2)
np.save('trained_params/weights3_p3.npy', weights3_p3)
np.save('trained_params/weights4_p1.npy', weights4_p1)
np.save('trained_params/weights4_p2.npy', weights4_p2)
np.save('trained_params/biases3.npy', biases3)
np.save('trained_params/biases4.npy', biases4)
np.save('trained_params/biases5_p1.npy', biases5_p1)
np.save('trained_params/biases5_p2.npy', biases5_p2)
np.save('trained_params/biases5_p3.npy', biases5_p3)
np.save('trained_params/biases6_p1.npy', biases6_p1)
np.save('trained_params/biases6_p2.npy', biases6_p2)
