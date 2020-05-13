# import required libraries
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy import integrate

# specify the environment for two area power system
class TwoAreaPowerSystemEnv(gym.Env):
	"""
	A simulation of a two area power system for frequency control
	problems
	"""
	
	def __init__(self):
		"""
		Initialise the environment
		"""
		
		# inheret env class methods
		super(TwoAreaPowerSystemEnv, self).__init__()

		############################################
		# Set model parameters
		############################################
		# Set model constants for area 1
		self.K_sg_1 = 1
		self.T_sg_1 = 0.08
		self.K_t_1 = 1
		self.T_t_1 = 0.3
		self.K_gl_1 = 120
		self.T_gl_1 = 20

    	# Set model constants for area 2
		self.K_sg_2 = 1
		self.T_sg_2 = 0.08
		self.R_2 = 2.4
		self.K_t_2 = 1
		self.T_t_2 = 0.3
		self.K_gl_2 = 120
		self.T_gl_2 = 20

    	# Synchronising coefficient on tie line
		self.T12 = 0.1
		############################################

		############################################
		# Define the observation space
		############################################
		# position 1 is 
		# position 2 is 
		# position 3 is 
		# position 4 is 
		# position 5 is 
		# position 6 is 
		# position 7 is 
		obs_high = np.ones(7)*np.finfo(np.float32).max
		self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
		############################################

		############################################
		# Define action space
		############################################
		# position 1 is control action for area 1
		# position 2 is control action for area 2
		act_high = np.ones(2)*np.finfo(np.float32).max
		self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
		############################################

		############################################
		# Define temporal characteristics
		############################################
		# set up the time for simulation
		self.t = None			# set up the timing
		self.t_delta = 0.01		# set up the time delta
		self.t_max = 30			# max simulation time

		# set up the state space
		self.state = None

		# Clean up the agent
		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def del_p_L_1_func(self, t):
		if (t < 1):
			del_p_L = 0.00
		else:
			del_p_L = 0.01
		return del_p_L

	def del_p_L_2_func(self, t):
		if (t < 1):
			del_p_L = 0.00
		else:
			del_p_L = 0.00
		return del_p_L
	

	def int_power_system_sim(self, x_sys, t,
							 control_sig_1, control_sig_2,                  # control sig
							 K_sg_1, T_sg_1, K_t_1, T_t_1, K_gl_1, T_gl_1,  # area one
							 K_sg_2, T_sg_2, K_t_2, T_t_2, K_gl_2, T_gl_2,  # area two
							 T12):                                          # tie line
	    # area 1 simulation
		x_2_dot = (1/T_sg_1)*(K_sg_1*control_sig_1 - x_sys[0])
		x_3_dot = (1/T_t_1)*(K_t_1*x_sys[0] - x_sys[1])
		x_4_dot = (K_gl_1/T_gl_1)*(x_sys[1] - x_sys[3] - self.del_p_L_1_func(t)) - (1/T_gl_1)*x_sys[2]

	    # tie line simulation
		x_5_dot = 2*np.pi*T12*(x_sys[2] - x_sys[6])

	    # area 2 simulation
		x_7_dot = (1/T_sg_2)*(K_sg_2*control_sig_2 - x_sys[4])
		x_8_dot = (1/T_t_2)*(K_t_2*x_sys[4] - x_sys[5])
		x_9_dot = (K_gl_2/T_gl_2)*(x_sys[5] + x_sys[3] - self.del_p_L_2_func(t)) - (1/T_gl_2)*x_sys[6]

		return (x_2_dot, x_3_dot, x_4_dot, x_5_dot, x_7_dot, x_8_dot, x_9_dot)


	def step(self, action):
		"""
		Step the system forward by a single time step
		"""
		
		# store the received control signals
		control_sig_1, control_sig_2 = action


		# create the argument tuple
		arg_sys = (control_sig_1, control_sig_2,
				   self.K_sg_1, self.T_sg_1, self.K_t_1, self.T_t_1, self.K_gl_1, self.T_gl_1,
				   self.K_sg_2, self.T_sg_2, self.K_t_2, self.T_t_2, self.K_gl_2, self.T_gl_2,
				   self.T12)

		# step the ode system forward in time once pdate the state
		self.state = integrate.odeint(self.int_power_system_sim,
									  self.state,
									  np.array([self.t, self.t + self.t_max]),
									  args=arg_sys)

		# step time forward
		self.t += self.t_delta

		# set the done status if the time period has elapsed
		done = (self.t > self.t_max)
		done = bool(done)

		# provide the reward signal
		if not done:
			reward = 0.1


		return np.array(self.state), reward, done, {}



	############################################
	# Reset the agent
	############################################
	def reset(self):
		self.state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		return np.array(self.state)