#!/usr/bin/env python

# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.ndimage.filters import uniform_filter1d
from tikzplotlib import save as tikz_save

def main():

    # Import pickle resuts
    with open('plot_10000.pkl', 'rb') as f:
        training_reward = pickle.load(f)

    # Create the episode list
    episode = list(range(100,10000,10))

    # Create the average list for plotting
    avg_list = [np.mean(training_reward[(i-100):i]) for i in episode]

    # Smooth the average list
    avg_list_smoothed = uniform_filter1d(avg_list, 1)

    # Create the standard deviation list
    stdev_list = [np.std(training_reward[(i-100):i]) for i in episode]
    stdev_list_smoothed = uniform_filter1d(stdev_list, 1)

    # Plot the episodic raw reward
    plt.plot(episode, training_reward[100:10000:10])
    tikz_save('raw_reward_plot.tikz')
    plt.close()

    # Plot the average reward for and standard deviation band
    plt.plot(episode, avg_list)
    plt.fill_between(episode,
                     avg_list_smoothed + stdev_list_smoothed,
                     avg_list_smoothed - stdev_list_smoothed,
                     alpha=0.3,
                     edgecolor="#ff7f0e")

    tikz_save('average_reward_plot.tikz')
    plt.close()



if __name__ == '__main__':
    main()