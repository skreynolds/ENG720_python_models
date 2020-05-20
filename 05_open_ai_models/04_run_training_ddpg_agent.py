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

# import the run function
from train.ddpg_train import ddpg_train


# main function
def main():
	
	# spin up environment
	env = TwoAreaPowerSystemEnv()
	env.seed(2)

	# spin up agent
	agent = DdpgController(state_size=7, action_size=2, random_seed=2)

	# REMOVE IF NOT CONTINUING TRAINING
	# Load the actor and critic networks
	#agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
	#agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

	# spin up the power demand signal
	signal = StepSignal()

	# train the agent
	scores = ddpg_train(env, agent, signal)

	# plot the outcome from training the agent
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(1, len(scores)+1), scores)
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.show()


if __name__ == '__main__':
	main()