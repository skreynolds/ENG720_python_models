
#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Set model inputs
del_p_C = 0 # Reference change

# Set model constants
K_sg = 1
T_sg = 0.2
R = 1/20
K_t = 1
T_t = 0.5
K_gl = 1
T_gl = 10

# Set timesteps for sinulation to integrate over
delta_t = 0.01 # time step size (seconds)
t_max = 11 # max sim time (seconds)

t = np.linspace(0, t_max, t_max/delta_t)

# Set variable initial conditions
x_1_0 = 0.0
x_2_0 = 0.0
x_3_0 = 0.0

x_init = (x_1_0, x_2_0, x_3_0)

# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_func(t):
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.02
    return del_p_L

# Setting up first order model to undertake simulation with odeint
def int_power_system_sim(x_init, t, K_sg=1, T_sg=0.2,
                         R=1/20, K_t=1, T_t=0.5,
                         K_gl=1, T_gl=10, del_p_C=0):
    x_1_dot = (K_sg/T_sg)*del_p_C - (K_sg/(R*T_sg))*x_init[2] - (1/T_sg)*x_init[0]
    x_2_dot = (K_t/T_t)*x_init[0] - (1/T_t)*x_init[1]
    x_3_dot = (K_gl/T_gl)*x_init[1] - (K_gl/T_gl)*del_p_L_func(t) - (0.8/T_gl)*x_init[2]
    return x_1_dot, x_2_dot, x_3_dot

def main():
    x_vals_int = integrate.odeint(int_power_system_sim, x_init, t)

    plt.plot(t, x_vals_int[:,2])
    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')
    plt.show()


# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()