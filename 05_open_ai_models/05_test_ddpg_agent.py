#!/usr/bin/env python

# NOTE: using this in an environment that has access to gym

# import required libraries
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# import the environment
from env.TwoAreaPowerSystemEnv import TwoAreaPowerSystemEnv

# import the ddpg agent
from agent.DdpgController import DdpgController

# import the power demand signal
from demand.Demand import StepSignal

# main function
def main():
	
	# spin up environment
	env = TwoAreaPowerSystemEnv()
	env.seed(2)

	# spin up agent
	agent = DdpgController(state_size=7, action_size=2, random_seed=2)

	# spin up the power demand signal
	signal = StepSignal()

	# Load the actor and critic networks
	agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
	agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

	state = env.reset()

	# initialise empty list to store simulation output
	out_s_1 = [0]
	out_s_2 = [0]
	control_s_1 = [0]
	control_s_2 = [0]
	time = [0]

	score = 0

	while True:
		
		action = agent.act(state, add_noise=False)

		demand = (signal.del_p_L_1_func(env.t),
				  signal.del_p_L_2_func(env.t))

		state, reward, done, _ = env.step(action, demand)

		score += reward

		out_s_1.append(state[2])
		out_s_2.append(state[6])
		time.append(env.t)
		control_s_1.append(action[0])
		control_s_2.append(action[1])

		if done:
			break

	print('Score: {}'.format(score))

	plt.subplot(311)
	plt.plot(time, out_s_1)
	plt.plot(time, out_s_2)

	plt.subplot(312)
	plt.plot(time, control_s_1)

	plt.subplot(313)
	plt.plot(time, control_s_2)

	plt.show()

if __name__ == '__main__':
	main()