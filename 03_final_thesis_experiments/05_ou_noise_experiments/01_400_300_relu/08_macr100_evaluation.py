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
    episode = list(range(100,10010,10))

    # Create the average list for plotting
    avg_list = np.asarray([np.mean(training_reward[(i-100):i]) for i in episode])

    print(len(avg_list))
    print(len(episode))
    print('The MACR100 at episode {}: {}'.format(episode[-1], avg_list[-1]))
    print('The best MACR100 during training is {} at episode {}'.
        format(max(avg_list), episode[np.argmax(avg_list)]))



if __name__ == '__main__':
    main()