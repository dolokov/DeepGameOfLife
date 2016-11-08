""" 
definition of update rules

convolve with (3,3) one kernel to sum alive neighbours

If the cell is alive, but has one or zero neighbours, it dies through under-population.
If the cell is alive and has two or three neighbours, it stays alive.
If the cell has more than three neighbours it dies through over-population.
Any dead cell with three neighbours regenerates.

"""

import numpy as np
from scipy.signal import convolve2d

def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X
