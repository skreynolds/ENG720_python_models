# import required libraries
import numpy as np
import random


# Specify the signal
class StepSignal():

    def __init__(self):
        """
        Initialise the signal class
        """

        # initialise the step transition size
        self.step_transition = None

        # initialise demand signal for area one
        self.area_one = None
        self.step_size_one = None

        # initialise demand signal for area two
        self.area_two = None
        self.step_size_two = None

        # initialise current signal
        self.current_sig = None


    def del_p_L_1_func(self, t):

        if self.area_one == 'on':
            if t < self.step_transition:
                del_p_L = 0.0
            else:
                del_p_L = self.step_size_one
        else:
            del_p_L = 0.0

        return del_p_L


    def del_p_L_2_func(self, t):
        if self.area_two == 'on':
            if t < self.step_transition:
                del_p_L = 0.0
            else:
                del_p_L = self.step_size_two
        else:
            del_p_L = 0.0

        return del_p_L


    def reset(self,
              step_transition,
              area_one, step_size_one,
              area_two, step_size_two):

        # initialise the step transition size
        self.step_transition = step_transition

        # initialise demand signal for area one
        self.area_one = area_one
        self.step_size_one = step_size_one

        # initialise demand signal for area two
        self.area_two = area_two
        self.step_size_two = step_size_two

        # initialise stochastic demand signal
        self.current_sig = 0.0


    def stochastic_signal_step(self,
                               increment=0.001,
                               min_sig=-0.01,
                               max_sig=0.01):
        # create random number
        rand_float = (random.random() - 0.5)/0.5

        # update current signal
        self.current_sig += rand_float*increment

        # check boundries
        if self.current_sig > max_sig:
            self.current_sig = max_sig
        elif self.current_sig < min_sig:
            self.current_sig = min_sig

        return self.current_sig