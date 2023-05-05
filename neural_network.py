import numpy as np
import pdb
from scipy.signal import convolve
import math


class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self, filters1, biases1, filters2, biases2):
        self.filters1 = filters1
        self.filters2 = filters2
        self.biases1 = biases1
        self.biases2 = biases2
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


    def query_cnn(self, map_image): # cnn - convolutional neural network

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



    def query_ann(self): # ann - artifical/vanilla neural network
        return