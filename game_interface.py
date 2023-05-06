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

    def closest_tower(self, position):
        distances = np.array([])

        for x in self.tower_positions:
            distances = np.append(distances, self.distance(x, position))
        if np.size(distances) == 0:
            return False
        closest_tower_position = self.tower_positions[distances.argmin()]

        return closest_tower_position

    def position_to_coords(self, position):
        position = np.array(position, dtype=np.float64)
        coords = position * 4
        coords[0] += 24
        return coords



    def place_tower(self, position, tower):
        pyautogui.click(x=48, y=0)
        pyautogui.click(x=48, y=0)

        coordinates = self.position_to_coords(position)

        print(f"Place tower at {coordinates[0]} {coordinates[1]}")
        return

    def upgrade_tower(self, position, upgrade_path):
        pyautogui.click(x=48, y=0)
        pyautogui.click(x=48, y=0)

        coordinates = self.position_to_coords(position)

        closest_tower_position = self.closest_tower(position)
        closest_tower_coords = self.position_to_coords(closest_tower_position)

        print(f"Upgrade tower closest to coordinates {coordinates[0]} {coordinates[1]}")
        print(f"-------> Tower at: {closest_tower_coords[0]} {closest_tower_coords[1]}")
        return

    def sell_tower(self, position):
        pyautogui.click(x=48, y=0)
        pyautogui.click(x=48, y=0)

        coordinates = self.position_to_coords(position)

        closest_tower_position = self.closest_tower(position)
        closest_tower_coords = self.position_to_coords(closest_tower_position)

        print(f"Sell tower closest to coordinates {coordinates[0]} {coordinates[1]}")
        print(f"-------> Tower at: {closest_tower_coords[0]} {closest_tower_coords[1]}")
        return