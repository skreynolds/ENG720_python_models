#!/usr/bin/env python

# import required libraries
import gym
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding
from env.TwoAreaPowerSystemEnv import TwoAreaPowerSystemEnv
from agent.ClassicalPiController import ClassicalPiController
from demand.Demand import StepSignal
from scipy import integrate

# import utility functions
from train.utils import *

def agent_test(mag, dur):

    ####################################
    # Create the environment
    ####################################
    # spin up environment
    env = TwoAreaPowerSystemEnv()

    # reset the agent
    state = env.reset()
    ####################################

    # View the action space and the state space
    #print("Action space: {}".format(env.action_space))
    #print("State space: {}".format(env.observation_space))

    ####################################
    # Create the controller
    ####################################
    # implement controller
    agent = ClassicalPiController()

    # reset the controller
    action = agent.reset()
    ####################################

    ####################################
    # Create the signal
    ####################################
    # implement the signal
    signal = StepSignal()
    signal.reset(dur, 'on', mag, 'off', 0.0)
    ####################################

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

        # Obtain the current demand
        demand = (signal.del_p_L_1_func(env.t),		# power demand for area 1
                  signal.del_p_L_2_func(env.t))		# power demand for area 2

        # Step the environment forward by one step
        state, reward, done, _ = env.step(action, demand)

        score += reward

        out_s_1.append(state[2])
        out_s_2.append(state[6])
        out_tieline.append(state[3])
        demand_list.append(demand[0])
        time_list.append(env.t)

        # Given the current state observation take an action
        action = agent.act(state, (env.t, env.t + env.t_delta))

        control_s_1.append(action[0])
        control_s_2.append(action[1])

        action /= 0.02

        if done:
            break

    print('Score: {}'.format(score))

    if mag < 0:
        fp = 'neg' 
    else:
        fp = 'pos'

    png_plot_file_path = './test_plots/pi_test_plot/pngplot/{}_{}_plot_final.png'.format(fp,dur)
    pik_file_path = './test_plots/pi_test_plot/pickledata/{}_{}_plot_final.pkl'.format(fp,dur)
    capture_agent_progress(time_list,
                           out_s_1, out_s_2,
                           control_s_1, control_s_2,
                           out_tieline,
                           demand_list,
                           png_plot_file_path,
                           pik_file_path)

def main():

    stepchange_list = [0.01, -0.01]
    duration_list = [5, 10, 15, 20, 25]

    for magnitude in stepchange_list:
        for duration in duration_list:
            agent_test(magnitude, duration)

if __name__ == '__main__':
    main()