import numpy as np

tower_positions = np.array([]).reshape(0, 2)
np.save('tower_positions.npy', tower_positions)


# HOW TO APPEND A POSITION ---->
# tower_positions = np.append(tower_positions, np.array([[300, 100]]), axis=0)