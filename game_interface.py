import numpy as np
import pdb
import pyautogui
import math


class GameInterface:

    def __init__(self, tower_positions):
    	self.tower_positions = tower_positions
    	return

    def distance(self, point_1, point_2):
        return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    def closest_tower(self, coordinates):
        distances = np.array([])

        for x in self.tower_positions:
            distances = np.append(distances, self.distance(x, coordinates))
        if np.size(distances) == 0:
            return False
        chosen_tower_coords = self.tower_positions[distances.argmin()]

        return chosen_tower_coords


    def place_tower(self, coordinates, tower):
        pdb.set_trace()
        return

    def upgrade_tower(self, coordinates, upgrade_path):
        pdb.set_trace()
        return

    def sell_tower(self, coordinates):
        pdb.set_trace()
        return