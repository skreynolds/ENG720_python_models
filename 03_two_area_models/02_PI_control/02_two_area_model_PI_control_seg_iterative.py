
#Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

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

# Setting up first order model for power system
def int_power_system_sim(x_sys, t,
                         control_sig_1, control_sig_2,                  # control sig
                         K_sg_1, T_sg_1, K_t_1, T_t_1, K_gl_1, T_gl_1,  # area one
                         K_sg_2, T_sg_2, K_t_2, T_t_2, K_gl_2, T_gl_2,  # area two
                         T12):                                          # tie line
    # area 1 simulation
    x_2_dot = (1/T_sg_1)*(K_sg_1*control_sig_1 - x_sys[0])
    x_3_dot = (1/T_t_1)*(K_t_1*x_sys[0] - x_sys[1])
    x_4_dot = (K_gl_1/T_gl_1)*(x_sys[1] - x_sys[3] - del_p_L_1_func(t)) - (1/T_gl_1)*x_sys[2]

    # tie line simulation
    x_5_dot = 2*np.pi*T12*(x_sys[2] - x_sys[6])

    # area 2 simulation
    x_7_dot = (1/T_sg_2)*(K_sg_2*control_sig_2 - x_sys[4])
    x_8_dot = (1/T_t_2)*(K_t_2*x_sys[4] - x_sys[5])
    x_9_dot = (K_gl_2/T_gl_2)*(x_sys[5] + x_sys[3] - del_p_L_2_func(t)) - (1/T_gl_2)*x_sys[6]

    return x_2_dot, x_3_dot, x_4_dot, x_5_dot, x_7_dot, x_8_dot, x_9_dot

# Setting up first order model for controller
def int_control_system_sim(x_control_sys, t,
                           frequency_sig_1, frequency_sig_2,            # freq sig
                           tie_line_sig,                                # tie line sig
                           R_1, K_i_1, b_1,                             # area one
                           R_2, K_i_2, b_2):                            # area two

    # controller 1 simulation
    x_1_dot = K_i_1*(tie_line_sig + b_1*frequency_sig_1)

    # controller 2 simulation
    x_6_dot = K_i_2*(-tie_line_sig + b_2*frequency_sig_2)

    return x_1_dot, x_6_dot


##################################################################
# Main function that is executed when file called from terminal
##################################################################
def main():
    ###############################################
    # Initialise system parameters
    ###############################################

    # Set model constants for area 1
    K_i_1 = -0.671
    b_1 = 0.425
    K_sg_1 = 1
    T_sg_1 = 0.08
    R_1 = 2.4
    K_t_1 = 1
    T_t_1 = 0.3
    K_gl_1 = 120
    T_gl_1 = 20

    # Set model constants for area 2
    K_i_2 = -0.671
    b_2 = 0.425
    K_sg_2 = 1
    T_sg_2 = 0.08
    R_2 = 2.4
    K_t_2 = 1
    T_t_2 = 0.3
    K_gl_2 = 120
    T_gl_2 = 20

    # Synchronising coefficient on tie line
    T12 = 0.1
    ###############################################

    ###############################################
    # Initialise simulation
    ###############################################

    # Set timesteps for sinulation to integrate over
    delta_t = 0.01 # time step size (seconds)
    t_max = 30 # max sim time (seconds)

    # Set variable initial conditions
    x_1_0 = 0.0     # x_control_sys[0]
    x_2_0 = 0.0     # x_sys[0]
    x_3_0 = 0.0     # x_sys[1]
    x_4_0 = 0.0     # x_freq_1
    x_5_0 = 0.0     # x_sys[3]
    x_6_0 = 0.0     # x_control_sys[1]
    x_7_0 = 0.0     # x_sys[4]
    x_8_0 = 0.0     # x_sys[5]
    x_9_0 = 0.0     # x_freq_2

    x_sys = (x_2_0, x_3_0, x_4_0, x_5_0, x_7_0, x_8_0, x_9_0)   # initialise sys
    x_freq_1 = x_4_0                                            # initialise freq_1
    x_freq_2 = x_9_0                                            # initialise freq_2
    x_tie_line = x_5_0                                          # initialise tieline
    x_control_sys = (x_1_0, x_6_0)                              # initialise int control input
    x_control = (x_1_0 - (1/R_1)*x_4_0, x_6_0 - (1/R_2)*x_9_0)  # initialise controller signal

    # initialise empty list to store simulation output
    out_s_1 = [x_freq_1]
    out_s_2 = [x_freq_2]

    # initialise the time
    t = 0
    time = [t]

    '''while t < t_max:'''
    for _ in range(10):

        # step foward in time
        t_step = t + delta_t
        print(x_control)
        ##############################################################
        # step the system simulation forward in time
        ##############################################################

        # arguement tuple
        arg_sys = (x_control[0], x_control[1],                    # control signals
                   K_sg_1, T_sg_1, K_t_1, T_t_1, K_gl_1, T_gl_1,  # area one
                   K_sg_2, T_sg_2, K_t_2, T_t_2, K_gl_2, T_gl_2,  # area two
                   T12)                                           # tie line

        # time step simulation
        x_sys_vals = integrate.odeint(int_power_system_sim,         # ode system
                                      x_sys,                        # initial cond
                                      np.array([t, t_step]),        # time step
                                      args=arg_sys)                 # model args
        
        # save the new init values for the system
        x_2 = x_sys_vals[1,0]
        x_3 = x_sys_vals[1,1]
        x_freq_1 = x_sys_vals[1,2]      # (area 1 frequency)
        x_tie_line = x_sys_vals[1,3]    # (tie line power)
        x_7 = x_sys_vals[1,4]
        x_8 = x_sys_vals[1,5]
        x_freq_2 = x_sys_vals[1,6]      # (area 2 frequency)

        # create new x_sys
        x_sys = (x_2, x_3, x_freq_1, x_tie_line, x_7, x_8, x_freq_2)
        ##############################################################
        print(x_sys)
        ##############################################################
        # step the control simulation forward in time
        ##############################################################

        # arguement tuple
        arg_control = (x_freq_1, x_freq_2,      # freq sig
                       x_tie_line,              # tie line sig
                       R_1, K_i_1, b_1,         # area one
                       R_2, K_i_2, b_2)

        # time step simulation
        x_control_vals = integrate.odeint(int_control_system_sim,   # ode system
                                          x_control_sys,            # initial cond
                                          np.array([t, t_step]),    # time step
                                          args=arg_control)         # model args
        
        # save the new init values for the controller
        x_control_sys = (x_control_vals[1,0], x_control_vals[1,1])

        # create new x_control
        x_control = (x_control_sys[0] - (1/R_1)*x_freq_1,
                     x_control_sys[1] - (1/R_2)*x_freq_2)
        ##############################################################

        # save the output
        out_s_1.append(x_freq_1)
        out_s_2.append(x_freq_2)

        # ste t to the next time step
        t = t_step
        time.append(t)

    plt.plot(time, out_s_1)
    plt.plot(time, out_s_2)
    plt.show()

# Execute the main() function if run from the terminal
if __name__ == '__main__':
    main()