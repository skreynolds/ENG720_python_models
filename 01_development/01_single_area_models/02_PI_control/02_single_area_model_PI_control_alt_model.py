
#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

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
t_max = 11 # max sim time (seconds)

t = np.linspace(0, t_max, t_max/delta_t)

# Set variable initial conditions
x_2_0 = 0.0 # x_init[0]
x_3_0 = 0.0 # x_init[1]
x_4_0 = 0.0 # x_init[2]
x_5_0 = 0.0 # x_init[3]

x_init = (x_2_0, x_3_0, x_4_0, x_5_0)

# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_func(t):
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.2
    return del_p_L

# Segregating the control function from the model (partially)
def x_control(x2_val, x5_val, R):
    return x2_val - (1/R)*x5_val

# Setting up first order model to undertake simulation with odeint
def int_power_system_sim(x_init, t, K_sg=1, T_sg=0.2,
                         R=1/20, K_t=1, T_t=0.5,
                         K_gl=1, T_gl=10, K_i=-7):
    x_2_dot = K_i*x_init[3]
    x_3_dot = (K_sg/T_sg)*x_control(x_init[0], x_init[3], R) - (1/T_sg)*x_init[1]
    x_4_dot = (K_t/T_t)*x_init[1] - (1/T_t)*x_init[2]
    x_5_dot = (K_gl/T_gl)*x_init[2] - (K_gl/T_gl)*del_p_L_func(t) - (0.8/T_gl)*x_init[3]
    return x_2_dot, x_3_dot, x_4_dot, x_5_dot

def main():
    x_vals_int = integrate.odeint(int_power_system_sim, x_init, t)

    plt.plot(t, x_vals_int[:,3])

    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')
    plt.show()


# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()