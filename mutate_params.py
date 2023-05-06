import numpy as np

filters1 = np.load('trained_params/filters1.npy')
filters2 = np.load('trained_params/filters2.npy')
biases1 = np.load('trained_params/biases1.npy')
biases2 = np.load('trained_params/biases2.npy')
weights1 = np.load('trained_params/weights1.npy')
weights2 = np.load('trained_params/weights2.npy')
weights3_p1 = np.load('trained_params/weights3_p1.npy')
weights3_p2 = np.load('trained_params/weights3_p2.npy')
weights3_p3 = np.load('trained_params/weights3_p3.npy')
weights4_p1 = np.load('trained_params/weights4_p1.npy')
weights4_p2 = np.load('trained_params/weights4_p2.npy')
biases3 = np.load('trained_params/biases3.npy')
biases4 = np.load('trained_params/biases4.npy')
biases5_p1 = np.load('trained_params/biases5_p1.npy')
biases5_p2 = np.load('trained_params/biases5_p2.npy')
biases5_p3 = np.load('trained_params/biases5_p3.npy')
biases6_p1 = np.load('trained_params/biases6_p1.npy')
biases6_p2 = np.load('trained_params/biases6_p2.npy')

mutation_rate = 1.0

filters1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 3, 3, 16))
filters2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 3, 16, 32))
biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (16, 1))
biases2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (32, 1))
weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (211200, 30))
weights2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (30, 4))
weights3_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 109350))
weights3_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 109350))
weights3_p3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 109350))
weights4_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (109350, 23))
weights4_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (109350, 3))
biases3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (30, 1))
biases4_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 1))
biases5_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (109350, 1))
biases5_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (109350, 1))
biases5_p3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (109350, 1))
biases6_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (23, 1))
biases6_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 1))

filters1 = filters1 + filters1_gradient
filters2 = filters2 + filters2_gradient
biases1 = biases1 + biases1_gradient
biases2 = biases2 + biases2_gradient
weights1 = weights1 + weights1_gradient
weights2 = weights2 + weights2_gradient
weights3_p1 = weights3_p1 + weights3_p1_gradient
weights3_p2 = weights3_p2 + weights3_p2_gradient
weights3_p3 = weights3_p3 + weights3_p3_gradient
weights4_p1 = weights4_p1 + weights4_p1_gradient
weights4_p2 = weights4_p2 + weights4_p2_gradient
biases3 = biases3 + biases3_gradient
biases4 = biases4 + biases4_gradient
biases5_p1 = biases5_p1 + biases5_p1_gradient
biases5_p2 = biases5_p2 + biases5_p2_gradient
biases5_p3 = biases5_p3 + biases5_p3_gradient
biases6_p1 = biases6_p1 + biases6_p1_gradient
biases6_p2 = biases6_p2 + biases6_p2_gradient

np.save('mutated_params/filters1.npy', filters1)
np.save('mutated_params/filters2.npy', filters2)
np.save('mutated_params/biases1.npy', biases1)
np.save('mutated_params/biases2.npy', biases2)
np.save('mutated_params/weights1.npy', weights1)
np.save('mutated_params/weights2.npy', weights2)
np.save('mutated_params/weights3_p1.npy', weights3_p1)
np.save('mutated_params/weights3_p2.npy', weights3_p2)
np.save('mutated_params/weights3_p3.npy', weights3_p3)
np.save('mutated_params/weights4_p1.npy', weights4_p1)
np.save('mutated_params/weights4_p2.npy', weights4_p2)
np.save('mutated_params/biases3.npy', biases3)
np.save('mutated_params/biases4.npy', biases4)
np.save('mutated_params/biases5_p1.npy', biases5_p1)
np.save('mutated_params/biases5_p2.npy', biases5_p2)
np.save('mutated_params/biases5_p3.npy', biases5_p3)
np.save('mutated_params/biases6_p1.npy', biases6_p1)
np.save('mutated_params/biases6_p2.npy', biases6_p2)