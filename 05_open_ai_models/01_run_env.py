#!/usr/bin/env python 

# import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding
from env.TwoAreaPowerSystemEnvWithSignal import TwoAreaPowerSystemEnvWithSignal
from scipy import integrate

# Setting up first order model for controller
def int_control_system_sim(x_control_sys, t,
                           frequency_sig_1, frequency_sig_2,            # freq sig
                           tie_line_sig,                                # tie line sig
                           R_1, K_i_1, b_1,                             # area one
                           R_2, K_i_2, b_2):                            # area two

    # controller 1 simulation
    x_1_dot = K_i_1*(tie_line_sig + b_1*frequency_sig_1)

    # controller 2 simulation
    x_6_dot = K_i_2*(-tie_line_sig + b_2*frequency_sig_2)

    return x_1_dot, x_6_dot


def main():
	
	# spin up environment
	env = TwoAreaPowerSystemEnvWithSignal()

	# reset the agent
	state = env.reset()

	# set controller parameters
	K_i_1 = -0.671
	b_1 = 0.425
	R_1 = 2.4

	K_i_2 = -0.671
	b_2 = 0.425
	R_2 = 2.4

	# set initial control parameters
	x_1_0 = 0.0     # x_control_sys[0]
	x_6_0 = 0.0     # x_control_sys[1]

	x_control_sys = (x_1_0, x_6_0)           # initialise int control input
	x_control = (x_1_0 - (1/R_1)*state[2],	 # initialise controller 1 signal
				 x_6_0 - (1/R_2)*state[6])   # initialise controller 2 signal

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

		arg_control = (state[2], state[6],		# freq signals
					   state[3],				# tieline
					   R_1, K_i_1, b_1,
					   R_2, K_i_2, b_2)

		# time step simulation
		x_control_vals = integrate.odeint(int_control_system_sim,   				# ode system
                                          x_control_sys,            				# initial cond
										  np.array([env.t, env.t + env.t_delta]),   # time step
                                          args=arg_control)         				# model args
        
        # save the new init values for the controller
		x_control_sys = (x_control_vals[1,0], x_control_vals[1,1])

        # create new x_control
		x_control = (x_control_sys[0] - (1/R_1)*state[2],
					 x_control_sys[1] - (1/R_2)*state[6])

	plt.plot(time, out_s_1)
	plt.plot(time, out_s_2)
	plt.show()



if __name__ == '__main__':
	main()