import numpy as np
import pdb
from scipy.signal import convolve
import math


class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self, tower_positions, filters1, biases1, filters2, biases2, weights1, weights2_p1, weights2_p2, weights2_p3, weights2_p4, weights2_p5, weights2_p6, biases3, biases4_p1, biases4_p2, biases4_p3, biases4_p4, biases4_p5, biases4_p6):
        self.tower_positions = tower_positions

        self.filters1 = filters1
        self.filters2 = filters2
        self.biases1 = biases1
        self.biases2 = biases2

        self.weights1 = weights1
        self.weights2_p1 = weights2_p1
        self.weights2_p2 = weights2_p2
        self.weights2_p3 = weights2_p3
        self.weights2_p4 = weights2_p4
        self.weights2_p5 = weights2_p5
        self.weights2_p6 = weights2_p6
        self.biases3 = biases3
        self.biases4_p1 = biases4_p1
        self.biases4_p2 = biases4_p2
        self.biases4_p3 = biases4_p3
        self.biases4_p4 = biases4_p4
        self.biases4_p5 = biases4_p5
        self.biases4_p6 = biases4_p6
        return
        
    def softmax(self, x):
        # Subtract the maximum value from each element to avoid overflow
        x = x - np.max(x)
        # Compute the exponentials of each element
        exp_x = np.exp(x)
        # Normalize by dividing each row by the sum of its elements
        return exp_x / np.sum(exp_x)

    def reLU(self, x):
        return np.maximum(0, x)

    def max_pool(self, x, filter_size, stride):
        mp_size = np.array(x.shape) / stride
        mp_size = np.array([math.ceil(mp_size[0]), math.ceil(mp_size[1])])
        mp = np.zeros(mp_size)

        for i1 in range(mp_size[0]):
            for i2 in range(mp_size[1]):
                selection1 = [i1 * stride, i2 * stride]
                selection2 = [selection1[0] + filter_size[0], selection1[1] + filter_size[1]]
                max_val = np.amax(x[selection1[0]:selection2[0], selection1[1]:selection2[1]])
                mp[i1, i2] = max_val
        return mp


    def position(self, layer_5):
        max_value = np.max(layer_5)
        position = np.unravel_index(np.argmax(layer_5, axis=None), (405, 270))
        return position



    def query_cnn(self, map_image): # cnn - convolutional neural network

        map_image = map_image / np.max(map_image)

        num_filters_l1 = self.filters1.shape[3]
        num_filters_l2 = self.filters2.shape[3]
        layer_1 = []
        layer_2 = []

        for i in range(num_filters_l1):
            convolved_image = convolve(map_image, self.filters1[:, :, :, i], mode='valid').reshape(268, 403)
            convolved_image = convolved_image + self.biases1[i, 0]
            convolved_image = self.reLU(convolved_image)

            max_pooled_image = self.max_pool(convolved_image, (2, 2), 2)
            layer_1.append(max_pooled_image)
        layer_1 = np.stack(layer_1, axis=2)

        for i in range(num_filters_l2):
            convolved_image = convolve(layer_1, self.filters2[:, :, :, i], mode='valid').reshape(132, 200)
            convolved_image = convolved_image + self.biases2[i, 0]
            convolved_image = self.reLU(convolved_image)

            max_pooled_image = self.max_pool(convolved_image, (2, 2), 2)
            layer_2.append(max_pooled_image)
        layer_2 = np.stack(layer_2, axis=2)
        layer_2 = layer_2 / np.max(layer_2)
        
        return layer_2



    def query_ann(self, layer_2): # ann - artifical/vanilla neural network
        layer_3 = self.weights1 * layer_2.flatten().reshape(211200, 1)
        layer_3 = np.sum(layer_3, axis=0)
        layer_3 = layer_3.reshape(48, 1) + self.biases3
        layer_3 = self.reLU(layer_3)

        layer_4_p1 = self.weights2_p1 * layer_3
        layer_4_p1 = np.sum(layer_4_p1, axis=0)
        layer_4_p1 = layer_4_p1.reshape(4, 1) + self.biases4_p1
        layer_4_p1 = self.reLU(layer_4_p1)
        
        action = layer_4_p1.argmax()
        if self.tower_positions.shape[0] == 0: # If there are no towers, the output must be action 0 (place tower)
            action = 0
        # 0: Place Tower
        # 1: Upgrade Tower
        # 2: Sell Tower
        # 3: Do Nothing

        if action == 0: # Place Tower
            layer_4_p2 = self.weights2_p2 * layer_3
            layer_4_p2 = np.sum(layer_4_p2, axis=0).reshape(109350, 1)

            layer_4_p3 = self.weights2_p3 * layer_3
            layer_4_p3 = np.sum(layer_4_p3, axis=0).reshape(23, 1)

            position = self.position(layer_4_p2)
            tower = layer_4_p3.argmax()
            return [action, position, tower]

        elif action == 1: # Upgrade Tower
            layer_4_p4 = self.weights2_p4 * layer_3
            layer_4_p4 = np.sum(layer_4_p4, axis=0).reshape(109350, 1)

            layer_4_p5 = self.weights2_p5 * layer_3
            layer_4_p5 = np.sum(layer_4_p5, axis=0).reshape(3, 1)

            position = self.position(layer_4_p4)
            upgrade_path = layer_4_p5.argmax()
            return [action, position, upgrade_path]

        elif action == 2: # Sell Tower
            layer_4_p6 = self.weights2_p6 * layer_3
            layer_4_p6 = np.sum(layer_4_p6, axis=0).reshape(109350, 1)

            position = self.position(layer_4_p6)
            return [action, position]

        elif action == 3: # Do Nothing
            return [action]