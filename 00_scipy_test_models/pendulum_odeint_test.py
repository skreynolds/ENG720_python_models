
# Import libraries required for modelling
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Set model constants
m = 1 # mass (kg)
L = 1 # length (m)
b = 0 # damping value (kg/m^2-s)
g = 9.81 # gravity (m/s^2)
delta_t = 0.02 # time step size (seconds)
t_max = 10 # max sim time (seconds)
theta1_0 = np.pi/2 # initial angle (radians)
theta2_0 = 0 # initial angular velocity (rad/s)
theta_init = (theta1_0, theta2_0)

# Get timesteps
t = np.linspace(0, t_max, t_max/delta_t)

# Setting up first order model to undertake simulation with odeint
def int_pendulum_sim(theta_init, t, L=1, m=1, b=0, g=9.81):
    theta_dot_1 = theta_init[1]
    theta_dot_2 = -b/m*theta_init[1] - g/L*np.sin(theta_init[0])
    return theta_dot_1, theta_dot_2

def main():
    # Run the simulation
    theta_vals_int = integrate.odeint(int_pendulum_sim, theta_init, t)

    # Plot the results
    plt.plot(t, theta_vals_int[:,0], 'b', label='theta(t)')
    plt.plot(t, theta_vals_int[:,1], 'g', label='omega(t)')
    plt.xlabel('time (seconds)')
    plt.ylabel('magnitude (radians)')
    plt.legend()
    plt.show()

# Execute the main() function if run from terminal
if __name__ == '__main__':
    main()