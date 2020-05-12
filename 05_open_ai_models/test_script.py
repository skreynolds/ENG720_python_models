#!/usr/bin/env python 

# import required libraries
from env.TwoAreaPowerSystemEnv import TwoAreaPowerSystemEnv

def main():
	
	# spin up environment
	env = TwoAreaPowerSystemEnv()

	
	print(env.action_space)
	print(env.observation_space)
	print(env.state)


if __name__ == '__main__':
	main()