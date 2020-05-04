
#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# import stochastic simulation
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

t = np.linspace(0, t_max, t_max/delta_t)

# create the grid to select the simulation value from
time_grid = t

# create the simulated demand signal
simulation = np.zeros(t.shape[0])

# time instances when load change occurs
#load_change = int(30/delta_t)              # every 30 seconds
#load_change = 1000                         # every 10 seconds
load_change = 500                           # every 05 seconds

# create simulation
for i in range(1, len(simulation)):
    if (i % load_change == 0) or (i == 100):
        simulation[i] = stochastic_signal_step(simulation[i-1])
    else:
        simulation[i] = simulation[i-1]


# Set variable initial conditions
x_1_0 = 0.0
x_2_0 = 0.0
x_3_0 = 0.0
x_4_0 = 0.0

x_init = (x_1_0, x_2_0, x_3_0, x_4_0)

# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_func(t):

    idx = np.digitize(t, time_grid)

    '''
    if (t < 1):
        del_p_L = 0.00
    elif (1 <= t < 5):
        del_p_L = 0.2
    else:
        del_p_L = -0.2
    '''
    return simulation[idx-1]

# Setting up first order model to undertake simulation with odeint
def int_power_system_sim(x_init, t, K_sg=1, T_sg=0.2,
                         R=1/20, K_t=1, T_t=0.5,
                         K_gl=1, T_gl=10, K_i=-7):
    x_1_dot = K_i*x_init[3]
    x_2_dot = (K_sg/T_sg)*(x_init[0] - (1/R)*x_init[3]) - (1/T_sg)*x_init[1]
    x_3_dot = (K_t/T_t)*x_init[1] - (1/T_t)*x_init[2]
    x_4_dot = (K_gl/T_gl)*x_init[2] - (K_gl/T_gl)*del_p_L_func(t) - (0.8/T_gl)*x_init[3]
    return x_1_dot, x_2_dot, x_3_dot, x_4_dot

def main():
    x_vals_int = integrate.odeint(int_power_system_sim, x_init, t)

    plt.subplot(1,2,1)
    plt.plot(t, x_vals_int[:,3])
    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')

    plt.subplot(1,2,2)
    plt.plot(t,simulation)
    plt.xlabel('time (seconds)')

    plt.show()


# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()