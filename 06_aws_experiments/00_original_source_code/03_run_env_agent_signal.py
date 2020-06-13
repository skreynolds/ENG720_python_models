#!/usr/bin/env python

# import required libraries
import gym
import matplotlib
plt.use("Agg")

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


def main():

	####################################
	# Create the environment
	####################################
	# spin up environment
	env = TwoAreaPowerSystemEnv()

	# reset the agent
	state = env.reset()
	####################################

	# View the action space and the state space
	print("Action space: {}".format(env.action_space))
	print("State space: {}".format(env.observation_space))

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
	signal.reset(1, 'on', 0.01, 'off', 0.0)
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

		if done:
			break

	print('Score: {}'.format(score))

	png_plot_file_path = './test_plots/pi_test_plot/pngplot/zz_plot_final.png'
	pik_file_path = './test_plots/pi_test_plot/pickledata/zz_plot_final.pkl'
	capture_agent_progress(time_list,
						   out_s_1, out_s_2,
						   control_s_1, control_s_2,
						   out_tieline,
						   demand_list,
						   png_plot_file_path,
						   pik_file_path)


if __name__ == '__main__':
	main()