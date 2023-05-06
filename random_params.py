import numpy as np

# Creates a random set of filter values
filters1 = np.random.uniform(-1.0, 1.0, (3, 3, 3, 16)) # Convolution layer 1
filters2 = np.random.uniform(-1.0, 1.0, (3, 3, 16, 32)) # Convolution layer 2

# Creates a set of biases, all 0 to start
biases1 = np.full(16, 0, dtype=np.float64).reshape(16, 1) # Convolution layer 1
biases2 = np.full(32, 0, dtype=np.float64).reshape(32, 1) # Convolution layer 2

# Creates weights for the feed forward neural network
weights1 = np.random.uniform(-1.0, 1.0, (211200, 48))
weights2_p1 = np.random.uniform(-1.0, 1.0, (48, 4))
weights2_p2 = np.random.uniform(-1.0, 1.0, (48, 109350))
weights2_p3 = np.random.uniform(-1.0, 1.0, (48, 23))
weights2_p4 = np.random.uniform(-1.0, 1.0, (48, 109350))
weights2_p5 = np.random.uniform(-1.0, 1.0, (48, 3))
weights2_p6 = np.random.uniform(-1.0, 1.0, (48, 109350))

# Creates biases for the feed forward neural network
biases3 = np.full(48, 0, dtype=np.float64).reshape(48, 1) # Feed Forward layer 1
biases4_p1 = np.full(4, 0, dtype=np.float64).reshape(4, 1) # Feed Forward layer 2.1
biases4_p2 = np.full(109350, 0, dtype=np.float64).reshape(109350, 1) # Feed Forward layer 2.1
biases4_p3 = np.full(23, 0, dtype=np.float64).reshape(23, 1) # Feed Forward layer 2.1
biases4_p4 = np.full(109350, 0, dtype=np.float64).reshape(109350, 1) # Feed Forward layer 2.1
biases4_p5 = np.full(3, 0, dtype=np.float64).reshape(3, 1) # Feed Forward layer 2.1
biases4_p6 = np.full(109350, 0, dtype=np.float64).reshape(109350, 1) # Feed Forward layer 2.1

np.save('trained_params/filters1.npy', filters1)
np.save('trained_params/filters2.npy', filters2)
np.save('trained_params/biases1.npy', biases1)
np.save('trained_params/biases2.npy', biases2)

np.save('trained_params/weights1.npy', weights1)
np.save('trained_params/weights2_p1.npy', weights2_p1)
np.save('trained_params/weights2_p2.npy', weights2_p2)
np.save('trained_params/weights2_p3.npy', weights2_p3)
np.save('trained_params/weights2_p4.npy', weights2_p4)
np.save('trained_params/weights2_p5.npy', weights2_p5)
np.save('trained_params/weights2_p6.npy', weights2_p6)
np.save('trained_params/biases3.npy', biases3)
np.save('trained_params/biases4_p1.npy', biases4_p1)
np.save('trained_params/biases4_p2.npy', biases4_p2)
np.save('trained_params/biases4_p3.npy', biases4_p3)
np.save('trained_params/biases4_p4.npy', biases4_p4)
np.save('trained_params/biases4_p5.npy', biases4_p5)
np.save('trained_params/biases4_p6.npy', biases4_p6)
