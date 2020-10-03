#!/usr/bin/env python

# NOTE: using this in an environment that has access to gym

# import required libraries
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import gym
import random
import pickle
import torch
import numpy as np
from collections import deque

# import the environment
from env.TwoAreaPowerSystemEnv import TwoAreaPowerSystemEnv

# import the ddpg agent
from agent.DdpgController import DdpgController

# import the power demand signal
from demand.Demand import StepSignal

# import utility functions
from train.utils import *


# agent test function
def agent_test(mag, dur):

    # spin up environment
    env = TwoAreaPowerSystemEnv()
    env.seed(2)
    state = env.reset()

    # spin up agent
    agent = DdpgController(state_size=7, action_size=2, random_seed=2)

    # spin up the power demand signal
    signal = StepSignal()
    signal.reset(dur , 'on', mag , 'off', 0.0)

    # Load the actor and critic networks
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    # initialise empty list to store simulation output
    out_s_1 = [0]
    out_s_2 = [0]
    out_tieline = [0]
    control_s_1 = [0]
    control_s_2 = [0]
    demand_list = [0]
    time_list = [0]

    score = 0

    while True:

        action = agent.act(state, add_noise=False)

        demand = (signal.del_p_L_1_func(env.t),
                  signal.del_p_L_2_func(env.t))

        state, reward, done, _ = env.step(action, demand)

        score += reward

        out_s_1.append(state[2])
        out_s_2.append(state[6])
        out_tieline.append(state[3])
        control_s_1.append(action[0])
        control_s_2.append(action[1])
        demand_list.append(demand[0])
        time_list.append(env.t)

        if done:
            break

    print('Score: {}'.format(score))

    # Plot the agent performance png files
    if mag < 0:
        fp = 'neg' 
    else:
        fp = 'pos'
    
    png_plot_file_path = './test_plots/ddpg_test_plot/pngplot/{}_{}_plot_final.png'.format(fp,dur)
    pik_file_path = './test_plots/ddpg_test_plot/pickledata/{}_{}_plot_final.pkl'.format(fp,dur)
    capture_agent_progress(time_list,
                           out_s_1, out_s_2,
                           control_s_1, control_s_2,
                           out_tieline,
                           demand_list,
                           png_plot_file_path,
                           pik_file_path)


# main function
def main():
    
    stepchange_list = [0.01, -0.01]
    duration_list = [5, 10, 15, 20, 25]

    for magnitude in stepchange_list:
        for duration in duration_list:
            agent_test(magnitude, duration)


# call to main function
if __name__ == '__main__':
	main()