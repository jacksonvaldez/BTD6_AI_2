import numpy as np
import pdb
import pyautogui
import math
import time


class GameInterface:

    def __init__(self, tower_positions):
        self.tower_positions = tower_positions
        self.tower_names = ['Dart Monkey', 'Boomerang Monkey', 'Bomb Shooter', 'Tack Shooter', 'Ice Monkey', 'Glue Gunner', 'Sniper Monkey', 'Monkey Sub', 'Monkey Buccaneer', 'Monkey Ace', 'Heli Pilot', 'Mortar Monkey', 'Dartling Gunner', 'Wizard Monkey', 'Super Monkey', 'Ninja Monkey', 'Alchemist', 'Druid', 'Banana Farm', 'Spike Factory', 'Monkey Village', 'Engineer Monkey', 'Beast Handler']
        self.path_names = ['Top Path', 'Middle Path', 'Bottom Path']
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

        print(f"Place a(n) {self.tower_names[tower]} at {coordinates[0]} {coordinates[1]}")

        if tower <= 10:
            pyautogui.click(x=1770, y=950)
            pyautogui.scroll(50)
        else:
            pyautogui.click(x=1770, y=950)
            pyautogui.scroll(-50)
        time.sleep(0.5)

        x_clicks = [1710, 1830]
        y_clicks = [210, 340, 475, 610, 740, 880]

        tower = int(tower + 1)
        x_click = x_clicks[tower % 2]
        y_click = y_clicks[(tower % 12) // 2] # The // division sign is the same as regular division but integer division
        
        pyautogui.click(x=x_click, y=y_click) # Select Tower
        pyautogui.click(x=coordinates[0], y=coordinates[1]) # Place Tower

        success = input("Did the tower place successfuly? (y/n) ")
        if success == 'y':
            self.tower_positions = np.append(self.tower_positions, np.array([position]), axis=0)
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