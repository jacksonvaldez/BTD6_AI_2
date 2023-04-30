import PIL.ImageGrab as ImageGrab
import pyautogui
import pdb
import numpy as np

left = 24
top = 0
width = 1620
height = 1080
region = (left, top, left+width, top+height)
screenshot = ImageGrab.grab(bbox=region) # Take a screenshot of the map
screenshot = screenshot.resize((420, 280)) # Resize the screenshot to 420 x 280
screenshot = screenshot.convert('L') # Convert the screenshot to grayscale
screenshot.save("screenshot.png") # Save the screenshot to a file
screenshot = np.array(screenshot, dtype=np.float64).flatten() / 255 # Convert the screenshot to a flattened numpy array with values scaled to 0 - 1