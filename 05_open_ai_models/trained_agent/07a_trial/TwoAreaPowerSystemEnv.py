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
    Observation space:
    Action space:
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
		self.K_sg_1, self.T_sg_1 = 1, 0.08   # governor parameters
		self.K_t_1, self.T_t_1 = 1, 0.3      # turbine parameters
		self.K_gl_1, self.T_gl_1 = 120, 20   # generator-load parameters

    	# Set model constants for area 2
		self.K_sg_2, self.T_sg_2 = 1, 0.08   # governor parameters
		self.K_t_2, self.T_t_2 = 1, 0.3      # turbine parameters
		self.K_gl_2, self.T_gl_2 = 120, 20   # generator-load parameters

    	# Synchronising coefficient on tie line
		self.T12 = 0.1

		############################################
		# Define the observation space
		############################################
		obs_high = np.ones(7)*np.finfo(np.float32).max
		self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

		############################################
		# Define action space
		############################################
		act_high = np.ones(2)*np.finfo(np.float32).max
		self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)

		############################################
		# Define temporal characteristics
		############################################
		# set up the time for simulation
		self.t = None			# set up the timing
		self.t_delta = 0.01		# set up the time delta
		self.t_max = 30			# max simulation time

		# set up the state space
		self.state = None


	def seed(self, seed=None):
		"""
        Set the random seed
        """
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def int_power_system_sim(self, x_sys, t,
							 control_sig_1, control_sig_2,
							 power_demand_sig_1, power_demand_sig_2,
							 K_sg_1, T_sg_1, K_t_1, T_t_1, K_gl_1, T_gl_1,
							 K_sg_2, T_sg_2, K_t_2, T_t_2, K_gl_2, T_gl_2,
							 T12):
		"""
		Inputs
		control sig
		power demand
		power demand
		area one
		area two
		tie line
		"""

		# area 1 simulation
		x_2_dot = (1/T_sg_1)*(K_sg_1*control_sig_1 - x_sys[0])
		x_3_dot = (1/T_t_1)*(K_t_1*x_sys[0] - x_sys[1])
		x_4_dot = (K_gl_1/T_gl_1)*(x_sys[1] - x_sys[3] - power_demand_sig_1) - \
                        (1/T_gl_1)*x_sys[2]

	    # tie line simulation
		x_5_dot = 2*np.pi*T12*(x_sys[2] - x_sys[6])

	    # area 2 simulation
		x_7_dot = (1/T_sg_2)*(K_sg_2*control_sig_2 - x_sys[4])
		x_8_dot = (1/T_t_2)*(K_t_2*x_sys[4] - x_sys[5])
		x_9_dot = (K_gl_2/T_gl_2)*(x_sys[5] + x_sys[3] - power_demand_sig_2) - \
                        (1/T_gl_2)*x_sys[6]

		return x_2_dot, x_3_dot, x_4_dot, x_5_dot, x_7_dot, x_8_dot, x_9_dot


	def step(self, action, demand):
		"""
		Step the system forward by a single time step

        Inputs:
        - action:
        - demand:

        Outputs:
        - state:
        - reward:
        - done:
		"""

		# store the received control signals
		control_sig_1, control_sig_2 = action

		# store the received power demand signals
		power_demand_sig_1, power_demand_sig_2 = demand

		# create the argument tuple
		arg_sys = (control_sig_1, control_sig_2,
				   power_demand_sig_1, power_demand_sig_2,
				   self.K_sg_1, self.T_sg_1, self.K_t_1, self.T_t_1, self.K_gl_1, self.T_gl_1,
				   self.K_sg_2, self.T_sg_2, self.K_t_2, self.T_t_2, self.K_gl_2, self.T_gl_2,
				   self.T12)

		# step the ode system forward in time once pdate the state
		x_sys_vals = integrate.odeint(self.int_power_system_sim,
									  self.state,
									  np.array([self.t, self.t + self.t_delta]),
									  args=arg_sys)

		self.state = (x_sys_vals[1,0],
					  x_sys_vals[1,1],
					  x_sys_vals[1,2],	# state[2] frequency 1
					  x_sys_vals[1,3],	# state[3] tieline
					  x_sys_vals[1,4],
					  x_sys_vals[1,5],
					  x_sys_vals[1,6])	# state[6] frequency 2

		# step time forward
		self.t += self.t_delta

		# set the done status if the time period has elapsed
		done = (self.t > self.t_max) or (abs(self.state[2]) > 0.75) or (abs(self.state[6]) > 0.75)
		done = bool(done)

		# provide the reward signal
		
		if (abs(self.state[2]) > 0.75) or (abs(self.state[6]) > 0.75):
			reward = -50*(abs(self.state[2]) + abs(self.state[6]))
		else:
			reward = 0.2 - 10*( abs(self.state[2]) + abs(control_sig_1)
				  	 	   	  + abs(self.state[3]) + abs(control_sig_2)
				  	 	   	  + abs(self.state[6]) )
		
		return np.array(self.state), reward, done, {}



	############################################
	# Reset the agent
	############################################
	def reset(self):
		self.state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		self.t = 0
		return np.array(self.state)