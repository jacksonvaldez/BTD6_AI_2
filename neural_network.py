import numpy as np
import pdb
from scipy.signal import convolve
import math


class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self, filters1, biases1, filters2, biases2, weights1, weights2, weights3_p1, weights3_p2, weights3_p3, weights4_p1, weights4_p2, biases3, biases4, biases5_p1, biases5_p2, biases5_p3, biases6_p1, biases6_p2):
        self.filters1 = filters1
        self.filters2 = filters2
        self.biases1 = biases1
        self.biases2 = biases2

        self.weights1 = weights1
        self.weights2 = weights2
        self.weights3_p1 = weights3_p1
        self.weights3_p2 = weights3_p2
        self.weights3_p3 = weights3_p3
        self.weights4_p1 = weights4_p1
        self.weights4_p2 = weights4_p2
        self.biases3 = biases3
        self.biases4 = biases4
        self.biases5_p1 = biases5_p1
        self.biases5_p2 = biases5_p2
        self.biases5_p3 = biases5_p3
        self.biases6_p1 = biases6_p1
        self.biases6_p2 = biases6_p2
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
        
        return layer_2



    def query_ann(self, layer_2): # ann - artifical/vanilla neural network
        layer_3 = self.weights1 * layer_2.flatten().reshape(211200, 1)
        layer_3 = np.sum(layer_3, axis=0)
        layer_3 = layer_3.reshape(30, 1) + self.biases3
        layer_3 = self.reLU(layer_3)


        layer_4 = self.weights2 * layer_3
        layer_4 = np.sum(layer_4, axis=0)
        layer_4 = layer_4.reshape(4, 1) + self.biases4
        layer_4 = self.reLU(layer_4)
        
        action = layer_4.argmax()
        # 0: Place Tower
        # 1: Upgrade Tower
        # 2: Sell Tower
        # 3: Do Nothing

        if action == 0: # Place Tower
            layer_5 = self.weights3_p1 * layer_4
            layer_5 = np.sum(layer_5, axis=0)
            layer_5 = layer_5.reshape(109350, 1) + self.biases5_p1
            layer_5 = self.reLU(layer_5)
            layer_6 = self.weights4_p1 * layer_5
            layer_6 = np.sum(layer_6, axis=0)
            layer_6 = layer_6.reshape(23, 1) + self.biases6_p1
            position = self.position(layer_5)
            tower = layer_6.argmax()
            return [action, position, tower]

        elif action == 1: # Upgrade Tower
            layer_5 = self.weights3_p2 * layer_4
            layer_5 = np.sum(layer_5, axis=0)
            layer_5 = layer_5.reshape(109350, 1) + self.biases5_p2
            layer_5 = self.reLU(layer_5)
            layer_6 = self.weights4_p2 * layer_5
            layer_6 = np.sum(layer_6, axis=0)
            layer_6 = layer_6.reshape(3, 1) + self.biases6_p2
            position = self.position(layer_5)
            upgrade_path = layer_6.argmax()
            return [action, position, upgrade_path]

        elif action == 2: # Sell Tower
            layer_5 = self.weights3_p3 * layer_4
            layer_5 = np.sum(layer_5, axis=0)
            layer_5 = layer_5.reshape(109350, 1) + self.biases5_p3
            layer_5 = self.reLU(layer_5)
            position = self.position(layer_5)
            return [action, position]

        elif action == 3:
            return [action]