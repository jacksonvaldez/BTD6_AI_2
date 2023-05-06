from PIL import Image, ImageGrab
import pyautogui
import pdb
import numpy as np
import cv2
from neural_network import NeuralNetwork
from game_interface import GameInterface
import time

params_path = "mutated"

filters1 = np.load(f"{params_path}_params/filters1.npy")
filters2 = np.load(f"{params_path}_params/filters2.npy")
biases1 = np.load(f"{params_path}_params/biases1.npy")
biases2 = np.load(f"{params_path}_params/biases2.npy")

weights1 = np.load(f"{params_path}_params/weights1.npy")
weights2 = np.load(f"{params_path}_params/weights2.npy")
weights3_p1 = np.load(f"{params_path}_params/weights3_p1.npy")
weights3_p2 = np.load(f"{params_path}_params/weights3_p2.npy")
weights3_p3 = np.load(f"{params_path}_params/weights3_p3.npy")
weights4_p1 = np.load(f"{params_path}_params/weights4_p1.npy")
weights4_p2 = np.load(f"{params_path}_params/weights4_p2.npy")
biases3 = np.load(f"{params_path}_params/biases3.npy")
biases4 = np.load(f"{params_path}_params/biases4.npy")
biases5_p1 = np.load(f"{params_path}_params/biases5_p1.npy")
biases5_p2 = np.load(f"{params_path}_params/biases5_p2.npy")
biases5_p3 = np.load(f"{params_path}_params/biases5_p3.npy")
biases6_p1 = np.load(f"{params_path}_params/biases6_p1.npy")
biases6_p2 = np.load(f"{params_path}_params/biases6_p2.npy")

tower_positions = np.load('tower_positions.npy')

game_interface = GameInterface(tower_positions)
neural_net = NeuralNetwork(filters1, biases1, filters2, biases2, weights1, weights2, weights3_p1, weights3_p2, weights3_p3, weights4_p1, weights4_p2, biases3, biases4, biases5_p1, biases5_p2, biases5_p3, biases6_p1, biases6_p2)

for x in range(1000):
	time.sleep(5)
	
	screenshot = game_interface.take_screenshot()
	query_cnn = neural_net.query_cnn(screenshot)
	query_ann = neural_net.query_ann(query_cnn)

	if query_ann[0] == 0: # Place Tower
		action, position, tower = query_ann
		game_interface.place_tower(position, tower)

	elif query_ann[0] == 1: # Upgrade Tower
		action, position, upgrade_path = query_ann
		game_interface.upgrade_tower(position, upgrade_path)

	elif query_ann[0] == 2: # Sell Tower
		action, position = query_ann
		game_interface.sell_tower(position)

	elif query_ann[0] == 3: # Do Nothing
		print("Do Nothing")
		action = query_ann

np.save('tower_positions.npy', game_interface.tower_positions)