# import required libraries
import numpy as np


# Specify the signal
class StepSignal():

	def __init__(self,
				 area_one='on',
				 step_size_one=0.01,
				 area_two='off',
				 step_size_two=0.0):
		"""
		Initialise the signal class
		"""

		self.area_one = area_one
		self.step_size_one = step_size_one

		self.area_two = area_two
		self.step_size_two = step_size_two


	def del_p_L_1_func(self, t):
		
		if self.area_one == 'on':
			if t < 1:
				del_p_L = 0.0
			else:
				del_p_L = self.step_size_one
		else:
			del_p_L = 0.0
		
		return del_p_L


	def del_p_L_2_func(self, t):
		if self.area_two == 'on':
			if t < 1:
				del_p_L = 0.0
			else:
				del_p_L = self.step_size_two
		else:
			del_p_L = 0.0
		
		return del_p_L