# import required libraries
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy import integrate

class ClassicalPiController():
	"""
	A classical controller for a two area power system
	NOTE: this controller is designed to be used with the TwoAreaPowerSystemEnv
	"""

	def __init__(self):
		"""
		Initialise the controller
		"""

		############################################
		# Set controller parameters
		############################################
		# set controller parameters for area one
		self.K_i_1 = -0.671
		self.b_1 = 0.425
		self.R_1 = 2.4

		# set controller parameters for area two
		self.K_i_2 = -0.671
		self.b_2 = 0.425
		self.R_2 = 2.4

		# initialise controller values
		self.controller_sys = (None, None)


	def int_control_system_sim(self, x_control_sys, t,
                           	   frequency_sig_1, frequency_sig_2,    # freq sig
                               tie_line_sig,                        # tie line sig
                               R_1, K_i_1, b_1,                     # area one
                               R_2, K_i_2, b_2):                    # area two

    	# controller 1 simulation
		x_1_dot = K_i_1*(tie_line_sig + b_1*frequency_sig_1)

    	# controller 2 simulation
		x_6_dot = K_i_2*(-tie_line_sig + b_2*frequency_sig_2)

		return x_1_dot, x_6_dot


	def act(self, state, t):
		"""
		Given the inputs the agent selects an action based on PI controller
		state	: a tuple (freq_1, freq_2, tieline)
		t 		: a tuple (current time step, current time step plus delta)
		"""
		freq_1, freq_2, tieline = state

		t, t_step = t

		arg_control = (freq_1, freq_2,		# freq signals
					   tieline,				# tieline
					   self.R_1, self.K_i_1, self.b_1,
					   self.R_2, self.K_i_2, self.b_2)

		# time step simulation
		x_control_vals = integrate.odeint(self.int_control_system_sim,    # ode system
                                          self.controller_sys,            # initial cond
										  np.array([t, t_step]),   		  # time step
                                          args=arg_control)         	  # model args
        
        # update the state of the controller
		self.controller_sys = (x_control_vals[1,0],
							   x_control_vals[1,1])

        # create action tuple
		action = (self.controller_sys[0] - (1/self.R_1)*freq_1,		# control action 1
				  self.controller_sys[1] - (1/self.R_2)*freq_2)		# control action 2

		return np.array(action)


	def reset(self):
		self.controller_sys = (0.0, 0.0)
		return np.array(self.controller_sys)