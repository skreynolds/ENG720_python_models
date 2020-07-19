
#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Set model constants for area 1
K_sg_1 = 1
T_sg_1 = 0.08
R_1 = 2.4
K_t_1 = 1
T_t_1 = 0.3
K_gl_1 = 120
T_gl_1 = 20

# Set model constants for area 2
K_sg_2 = 1
T_sg_2 = 0.08
R_2 = 2.4
K_t_2 = 1
T_t_2 = 0.3
K_gl_2 = 120
T_gl_2 = 20

# Synchronising coefficient on tie line
T12 = 0.1

# Set timesteps for sinulation to integrate over
delta_t = 0.01 # time step size (seconds)
t_max = 10 # max sim time (seconds)

t = np.linspace(0, t_max, t_max/delta_t)

# Set variable initial conditions
x_1_0 = 0.0
x_2_0 = 0.0
x_3_0 = 0.0
x_4_0 = 0.0
x_5_0 = 0.0
x_6_0 = 0.0
x_7_0 = 0.0

x_init = (x_1_0, x_2_0, x_3_0, x_4_0, x_5_0, x_6_0, x_7_0)

# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_1_func(t):
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.01
    return del_p_L

def del_p_L_2_func(t):
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.00
    return del_p_L

# Setting up first order model to undertake simulation with odeint
def int_power_system_sim(x_init, t,
                         K_sg_1=1, T_sg_1=0.08, R_1=2.4, K_t_1=1,
                         T_t_1=0.3, K_gl_1=120, T_gl_1=20,
                         K_sg_2=1, T_sg_2=0.08, R_2=2.4, K_t_2=1,
                         T_t_2=0.3, K_gl_2=120, T_gl_2=20,
                         T12=0.1):
    x_1_dot = (1/T_sg_1)*(K_sg_1*x_init[3] - (K_sg_1/R_1)*x_init[2] - x_init[0])
    x_2_dot = (1/T_t_1)*(K_t_1*x_init[0] - x_init[1])
    x_3_dot = (K_gl_1/T_gl_1)*(x_init[1] - x_init[3] - del_p_L_1_func(t)) - (1/T_gl_1)*x_init[2]
    x_4_dot = (2*np.pi*T12)*(x_init[2] - x_init[6])
    x_5_dot = (1/T_sg_2)*(-K_sg_2*x_init[3] - (K_sg_2/R_2)*x_init[6] - x_init[4])
    x_6_dot = (1/T_t_2)*(K_t_2*x_init[4] - x_init[5])
    x_7_dot = (K_gl_2/T_gl_2)*(x_init[5] + x_init[3] - del_p_L_2_func(t)) - (1/T_gl_2)*x_init[6]
    return x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_5_dot, x_6_dot, x_7_dot

def main():
    x_vals_int = integrate.odeint(int_power_system_sim, x_init, t)

    plt.plot(t, x_vals_int[:,2])
    plt.plot(t, x_vals_int[:,6])
    plt.show()

# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()