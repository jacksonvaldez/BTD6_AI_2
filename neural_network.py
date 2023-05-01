import numpy as np
import pdb
from scipy.signal import convolve


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


    def query_cnn(self, map_image):

        num_filters_l1 = self.filters1.shape[3]
        l1 = []

        for i in range(num_filters_l1):
            convolved_image = convolve(map_image, self.filters1[:, :, :, i], mode='valid').reshape(268, 403)
            convolved_image = convolved_image + self.biases1[i, 0]
            convolved_image = self.reLU(convolved_image)
            l1.append(convolved_image)

        l1 = np.stack(l1, axis=2)
        
        return

    def query_ann(self):
        return