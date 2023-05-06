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
weights2_p1 = np.load(f"{params_path}_params/weights2_p1.npy")
weights2_p2 = np.load(f"{params_path}_params/weights2_p2.npy")
weights2_p3 = np.load(f"{params_path}_params/weights2_p3.npy")
weights2_p4 = np.load(f"{params_path}_params/weights2_p4.npy")
weights2_p5 = np.load(f"{params_path}_params/weights2_p5.npy")
weights2_p6 = np.load(f"{params_path}_params/weights2_p6.npy")
biases3 = np.load(f"{params_path}_params/biases3.npy")
biases4_p1 = np.load(f"{params_path}_params/biases4_p1.npy")
biases4_p2 = np.load(f"{params_path}_params/biases4_p2.npy")
biases4_p3 = np.load(f"{params_path}_params/biases4_p3.npy")
biases4_p4 = np.load(f"{params_path}_params/biases4_p4.npy")
biases4_p5 = np.load(f"{params_path}_params/biases4_p5.npy")
biases4_p6 = np.load(f"{params_path}_params/biases4_p6.npy")

tower_positions = np.load('tower_positions.npy')

game_interface = GameInterface(tower_positions)
neural_net = NeuralNetwork(tower_positions, filters1, biases1, filters2, biases2, weights1, weights2_p1, weights2_p2, weights2_p3, weights2_p4, weights2_p5, weights2_p6, biases3, biases4_p1, biases4_p2, biases4_p3, biases4_p4, biases4_p5, biases4_p6)

for x in range(1000):
	time.sleep(5)
	
	neural_net.tower_positions = game_interface.tower_positions
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