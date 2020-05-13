#!/usr/bin/env python 

# import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding
from env.TwoAreaPowerSystemEnv import TwoAreaPowerSystemEnv
from scipy import integrate


def main():
	
	# spin up environment
	env = TwoAreaPowerSystemEnv()

	# reset the agent
	state = env.reset()
	
	print(env.action_space)
	print(env.observation_space)
	print(env.t)

	# implement controller

	# initialise empty list to store simulation output
	out_s_1 = [0]
	out_s_2 = [0]
	time = [0]

	while True:

		state, reward, done, _ = env.step(x_control)

		out_s_1.append(state[2])
		out_s_2.append(state[6])
		time.append(env.t)

		if done:
			break

		# implement controller

	plt.plot(time, out_s_1)
	plt.plot(time, out_s_2)
	plt.show()



if __name__ == '__main__':
	main()