#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from stochastic_signal import *

# Set model constants
K_sg = 1
T_sg = 0.2
R = 1/20
K_i = -7
K_t = 1
T_t = 0.5
K_gl = 1
T_gl = 10

# Set timesteps for sinulation to integrate over
delta_t = 0.01 # time step size (seconds)
t_max = 120 # max sim time (seconds)

# Initialise power demand list
power_demand_list = [0]

# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_func(t):
    '''
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.2
    '''
    return del_p_L

# Segregating the control function from the model (partially)
def x_control(x2_val, x5_val, R):
    return x2_val - (1/R)*x5_val

# Setting up first order model to undertake simulation with odeint
def int_power_system_sim(x_init, t, P, K_sg=1, T_sg=0.2,
                         R=1/20, K_t=1, T_t=0.5,
                         K_gl=1, T_gl=10, K_i=-7):
    x_2_dot = K_i*x_init[3]
    x_3_dot = (K_sg/T_sg)*x_control(x_init[0], x_init[3], R) - (1/T_sg)*x_init[1]
    x_4_dot = (K_t/T_t)*x_init[1] - (1/T_t)*x_init[2]
    x_5_dot = (K_gl/T_gl)*x_init[2] - (K_gl/T_gl)*P - (0.8/T_gl)*x_init[3]
    return x_2_dot, x_3_dot, x_4_dot, x_5_dot

def main():

    # Set variable initial conditions
    x_2_0 = 0.0 # x_init[0]
    x_3_0 = 0.0 # x_init[1]
    x_4_0 = 0.0 # x_init[2]
    x_5_0 = 0.0 # x_init[3]

    x_init = (x_2_0, x_3_0, x_4_0, x_5_0)

    # initialise empty list to store simulation output
    out = [x_5_0]

    # initialise the time
    t = 0
    time = [t]

    # Commence simulation
    while t < t_max:

        # step forward in time
        t_step = t + delta_t

        power_demand = power_demand_list[-1]

        # arguement tuple
        arg_tuple = (power_demand, K_sg, T_sg, R, K_t, T_t, K_gl, T_gl, K_i)

        # step the simulation forward in time
        x_vals_int = integrate.odeint(int_power_system_sim,     # ode system
                                      x_init,                   # initial conditions
                                      np.array([t, t_step]),    # time step
                                      args=arg_tuple)           # power demand

        # save the new init values
        x_2 = x_vals_int[1, 0] # x_init[0]
        x_3 = x_vals_int[1, 1] # x_init[1]
        x_4 = x_vals_int[1, 2] # x_init[2]
        x_5 = x_vals_int[1, 3] # x_init[3]

        # create new init
        x_init = (x_2, x_3, x_4, x_5)

        # save the output
        out.append(x_5)

        # set t to the next time step
        t = t_step
        time.append(t)

        # update the next power demand change
        if (round(t/delta_t)) % 3000 == 0: # every 3 seconds
            power_demand = stochastic_signal_step(power_demand)

        power_demand_list.append(power_demand)

    # plot the output
    plt.subplot(1,2,1)
    plt.plot(time, out)
    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')

    plt.subplot(1,2,2)
    plt.plot(time,np.array(power_demand_list))
    plt.xlabel('time (seconds)')
    plt.show()


# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()