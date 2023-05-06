from PIL import Image, ImageGrab
import pyautogui
import pdb
import numpy as np
import cv2
from neural_network import NeuralNetwork

left = 24
top = 0
width = 1620
height = 1080
region = (left, top, left+width, top+height)
screenshot = ImageGrab.grab(bbox=region) # Take a screenshot of the map
screenshot = screenshot.resize((405, 270))
screenshot = screenshot.convert("RGB") # Convert the screenshot to RGB format
screenshot.save("screenshot.png") # Save the screenshot to a file


screenshot = np.array(screenshot, dtype=np.float64) # Convert screenshot to numpy array

filters1 = np.load('trained_params/filters1.npy')
filters2 = np.load('trained_params/filters2.npy')
biases1 = np.load('trained_params/biases1.npy')
biases2 = np.load('trained_params/biases2.npy')

weights1 = np.load('trained_params/weights1.npy')
weights2 = np.load('trained_params/weights2.npy')
weights3_1 = np.load('trained_params/weights3_1.npy')
weights3_2 = np.load('trained_params/weights3_2.npy')
weights3_3 = np.load('trained_params/weights3_3.npy')
weights4_1 = np.load('trained_params/weights4_1.npy')
weights4_2 = np.load('trained_params/weights4_2.npy')
biases3 = np.load('trained_params/biases3.npy')
biases4 = np.load('trained_params/biases4.npy')
biases5_1 = np.load('trained_params/biases5_1.npy')
biases5_2 = np.load('trained_params/biases5_2.npy')
biases5_3 = np.load('trained_params/biases5_3.npy')
biases6_1 = np.load('trained_params/biases6_1.npy')
biases6_2 = np.load('trained_params/biases6_2.npy')

neural_net = NeuralNetwork(filters1, biases1, filters2, biases2, weights1, weights2, weights3_1, weights3_2, weights3_3, weights4_1, weights4_2, biases3, biases4, biases5_1, biases5_2, biases5_3, biases6_1, biases6_2)
query_cnn = neural_net.query_cnn(screenshot)
query_ann = neural_net.query_ann(query_cnn)