#!/usr/bin/env python 

# import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding
from env.TwoAreaPowerSystemEnvWithSignal import TwoAreaPowerSystemEnvWithSignal
from agent.ClassicalPiController import ClassicalPiController
from scipy import integrate


def main():
	
	####################################
	# Create the environment
	####################################
	# spin up environment
	env = TwoAreaPowerSystemEnvWithSignal()

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

	# initialise empty list to store simulation output
	out_s_1 = [0]
	out_s_2 = [0]
	time = [0]

	while True:

		# Step the environment forward by one step
		state, reward, done, _ = env.step(action)

		out_s_1.append(state[2])
		out_s_2.append(state[6])
		time.append(env.t)

		if done:
			break

		# Given the current state observation take an action
		action = agent.act(state, (env.t, env.t + env.t_delta))

	plt.plot(time, out_s_1)
	plt.plot(time, out_s_2)
	plt.show()



if __name__ == '__main__':
	main()