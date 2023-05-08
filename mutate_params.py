import numpy as np

filters1 = np.load('trained_params/filters1.npy')
filters2 = np.load('trained_params/filters2.npy')
biases1 = np.load('trained_params/biases1.npy')
biases2 = np.load('trained_params/biases2.npy')
weights1 = np.load('trained_params/weights1.npy')
weights2_p1 = np.load('trained_params/weights2_p1.npy')
weights2_p2 = np.load('trained_params/weights2_p2.npy')
weights2_p3 = np.load('trained_params/weights2_p3.npy')
weights2_p4 = np.load('trained_params/weights2_p4.npy')
weights2_p5 = np.load('trained_params/weights2_p5.npy')
weights2_p6 = np.load('trained_params/weights2_p6.npy')
biases3 = np.load('trained_params/biases3.npy')
biases4_p1 = np.load('trained_params/biases4_p1.npy')
biases4_p2 = np.load('trained_params/biases4_p2.npy')
biases4_p3 = np.load('trained_params/biases4_p3.npy')
biases4_p4 = np.load('trained_params/biases4_p4.npy')
biases4_p5 = np.load('trained_params/biases4_p5.npy')
biases4_p6 = np.load('trained_params/biases4_p6.npy')

mutation_rate = 0.3

filters1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 3, 3, 16))
filters2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 3, 16, 32))
biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (16, 1))
biases2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (32, 1))
weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (211200, 128))
weights2_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 4))
weights2_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 256))
weights2_p3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 23))
weights2_p4_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 256))
weights2_p5_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 3))
weights2_p6_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 256))
biases3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (128, 1))
biases4_p1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 1))
biases4_p2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (256, 1))
biases4_p3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (23, 1))
biases4_p4_gradient = np.random.uniform(-mutation_rate, mutation_rate, (256, 1))
biases4_p5_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 1))
biases4_p6_gradient = np.random.uniform(-mutation_rate, mutation_rate, (256, 1))

filters1 = filters1 + filters1_gradient
filters2 = filters2 + filters2_gradient
biases1 = biases1 + biases1_gradient
biases2 = biases2 + biases2_gradient
weights1 = weights1 + weights1_gradient
weights2_p1 = weights2_p1 + weights2_p1_gradient
weights2_p2 = weights2_p2 + weights2_p2_gradient
weights2_p3 = weights2_p3 + weights2_p3_gradient
weights2_p4 = weights2_p4 + weights2_p4_gradient
weights2_p5 = weights2_p5 + weights2_p5_gradient
weights2_p6 = weights2_p6 + weights2_p6_gradient
biases3 = biases3 + biases3_gradient
biases4_p1 = biases4_p1 + biases4_p1_gradient
biases4_p2 = biases4_p2 + biases4_p2_gradient
biases4_p3 = biases4_p3 + biases4_p3_gradient
biases4_p4 = biases4_p4 + biases4_p4_gradient
biases4_p5 = biases4_p5 + biases4_p5_gradient
biases4_p6 = biases4_p6 + biases4_p6_gradient

np.save('mutated_params/filters1.npy', filters1)
np.save('mutated_params/filters2.npy', filters2)
np.save('mutated_params/biases1.npy', biases1)
np.save('mutated_params/biases2.npy', biases2)
np.save('mutated_params/weights1.npy', weights1)
np.save('mutated_params/weights2_p1.npy', weights2_p1)
np.save('mutated_params/weights2_p2.npy', weights2_p2)
np.save('mutated_params/weights2_p3.npy', weights2_p3)
np.save('mutated_params/weights2_p4.npy', weights2_p4)
np.save('mutated_params/weights2_p5.npy', weights2_p5)
np.save('mutated_params/weights2_p6.npy', weights2_p6)
np.save('mutated_params/biases3.npy', biases3)
np.save('mutated_params/biases4_p1.npy', biases4_p1)
np.save('mutated_params/biases4_p2.npy', biases4_p2)
np.save('mutated_params/biases4_p3.npy', biases4_p3)
np.save('mutated_params/biases4_p4.npy', biases4_p4)
np.save('mutated_params/biases4_p5.npy', biases4_p5)
np.save('mutated_params/biases4_p6.npy', biases4_p6)