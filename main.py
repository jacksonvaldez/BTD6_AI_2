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

neural_net = NeuralNetwork(filters1, biases1, filters2, biases2)
query = neural_net.query_cnn(screenshot)
print('s')