#!/usr/bin/env python

"""
NOTE: this should be run from an environment with Python 3.6 Anaconda installed
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt

from stochastic_signal import *

# main function to generate and plot a simulated signal
def main():

    # set random seed
    random.seed(0)

    # specify the time over which we want to genearate the signal
    t_init = 0
    t_max = 11
    t_delta = 0.01

    # numpy array of time
    t = np.linspace(t_init, t_max, (t_max-t_init)/t_delta)

    # initialise numpy array to store simulation
    simulation = np.zeros(t.shape[0])

    for i in range(1, len(simulation)):
        simulation[i] = stochastic_signal_step(simulation[i-1])

    # plotting simulation output
    plt.plot(t, simulation)
    plt.xlabel('time (seconds)')
    plt.ylabel('P')
    plt.show()


if __name__ == '__main__':
    main()