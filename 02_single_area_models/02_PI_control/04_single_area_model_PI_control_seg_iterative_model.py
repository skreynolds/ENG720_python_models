#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# Setting up step function for power change
# Note that for the purposes of DRL this could be any signal
def del_p_L_func(t):
    if (t < 1):
        del_p_L = 0.00
    else:
        del_p_L = 0.2
    return del_p_L


# Setting up first order model for power system
def int_power_system_sim(x_sys, t, control_sig, K_sg=1, T_sg=0.2,
                         K_t=1, T_t=0.5, K_gl=1, T_gl=10):
    #print(control_sig)
    x_3_dot = (K_sg/T_sg)*control_sig - (1/T_sg)*x_sys[0]
    x_4_dot = (K_t/T_t)*x_sys[0] - (1/T_t)*x_sys[1]
    x_5_dot = (K_gl/T_gl)*x_sys[1] - (K_gl/T_gl)*del_p_L_func(t) - (0.8/T_gl)*x_sys[2]
    return x_3_dot, x_4_dot, x_5_dot


# Setting up first order model for controller
def int_control_system_sim(x_control, t, x_frequency, R=1/20, K_i=-7):
    x_2_dot = K_i*x_frequency
    return x_2_dot



def main():
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

    # Set variable initial conditions
    x_2_0 = 0.0 # x_control[0]  (system control signal)
    x_3_0 = 0.0 # x_sys[0]
    x_4_0 = 0.0 # x_sys[1]
    x_5_0 = 0.0 # x_sys[2]      (system frequency)

    x_sys = (x_3_0, x_4_0, x_5_0)       # initialise system
    x_2 = x_2_0                         # initialise integral control input
    x_control = x_2_0 - (1/R)*x_5_0     # initialise controller signal

    # initialise empty list to store simulation output
    out_s = [x_5_0]     # frequency output
    out_c = [x_2_0]    # controller output

    # initialise the time
    t = 0
    time = [t]

    while t < t_max:

        # step forward in time
        t_step = t + delta_t

        ##############################################################
        # step the system simulation forward in time
        # arguement tuple
        arg_sys = (x_control, K_sg, T_sg, K_t, T_t, K_gl, T_gl)
        #print(arg_sys)
        x_sys_vals = integrate.odeint(int_power_system_sim,         # ode system
                                      x_sys,                        # initial cond
                                      np.array([t, t_step]),        # time step
                                      args=arg_sys)                 # model args
        ##############################################################

        ##############################################################
        # step the control simulation forward in time
        arg_control = (x_sys[2], R, K_i)
        x_control_vals = integrate.odeint(int_control_system_sim,   # ode system
                                          x_2,                      # initial cond
                                          np.array([t, t_step]),    # time step
                                          args=arg_control)         # model args
        ##############################################################

        x_3 = x_sys_vals[1, 0] # x_sys[0]
        x_4 = x_sys_vals[1, 1] # x_sys[1]
        x_5 = x_sys_vals[1, 2] # x_sys[2] (system frequency)

        # create new x_sys
        x_sys = (x_3, x_4, x_5)

        # save the new init values
        x_2 = x_control_vals[1, 0] # x_init[0]

        # create new x_control
        x_control = x_2 - (1/R)*x_5

        # save the output
        out_s.append(x_5)
        out_c.append(x_2)

        # set t to the next time step
        t = t_step
        time.append(t)
        #print("Time: {}\t\tValue: {}".format(t,x_5))

    # plot the output
    ''' Plot frequency and control effort
    plt.subplot(121)
    plt.plot(time, out_s)
    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')

    plt.subplot(122)
    plt.plot(time, out_c)
    plt.xlabel('time (seconds)')
    '''

    plt.plot(time, out_s)
    plt.xlabel('time (seconds)')
    plt.ylabel('frequency (Hertz)')

    plt.show()


# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()