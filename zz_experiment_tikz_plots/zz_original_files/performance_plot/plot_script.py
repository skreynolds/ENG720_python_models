#!/usr/bin/env python

# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tikzplotlib import save as tikz_save

def main():

	# Import the pickle results
	with open('dump.pkl', 'rb') as f:
		pi_results = pickle.load(f)

	'''
	with open('ddpg_dump.pkl', 'rb') as f:
		ddpg_results = pickle.load(f)


	# Tikz plot the system response
	plt.plot(pi_results[0], pi_results[1])
	plt.plot(ddpg_results[0], ddpg_results[1])
	tikz_save('frequency_response_1.tikz')
	plt.close()

	# Tikz plot the system response
	plt.plot(pi_results[0], pi_results[2])
	plt.plot(ddpg_results[0], ddpg_results[2])
	tikz_save('frequency_response_2.tikz')
	plt.close()

	# Tikz plot the first control signal
	plt.plot(pi_results[0], pi_results[3])
	plt.plot(ddpg_results[0], [0.02*e for e in ddpg_results[3]])
	tikz_save('control_signal_1.tikz')
	plt.close()

	# Tikz plot the second control signal
	plt.plot(pi_results[0], pi_results[4])
	plt.plot(ddpg_results[0], [0.02*e for e in ddpg_results[4]])
	tikz_save('control_signal_2.tikz')
	plt.close()
	'''

	# Tikz plot the power demand step
	plt.plot(pi_results[0], pi_results[5])
	tikz_save('demand.tikz')
	plt.close()


if __name__ == '__main__':
	main()