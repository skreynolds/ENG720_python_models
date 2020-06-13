<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 02
## Intent
Experiment 1 highlighted that the system is very sensitive to chnages in control action. If the magnitude of the control change is too large then this can cause the controller difficulty arriving at an optimal policy.

This may be due to the the hyperbolic tangent function for the output of the actor network not being able to obtain sufficient selection of small values.

Experiment 1 also highlighted issues with the original specification of the reward function. This has now been modified to ensure that the agent will receive a positive reward for each time step, but loses part of that reward dependent on how far away the frequency is from the desired value of zero (i.e. not moving from the frequency set point).


## Hyperparameters
The specification of the hyperparamaters for the ddpg training algorithm are as follows:
```
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
WEIGHT_DECAY = 0
```

## Reward Function
The reward function for this experiment is defined as:
```python
if (abs(self.state[2]) > 0.75) or (abs(self.state[6]) > 0.75):
    reward = -2050*(abs(self.state[2]) + abs(self.state[6]))
else:
    reward = 1.2 - 0.1*( abs(self.state[2]) + 2*abs(control_sig_1)
                       + abs(self.state[3]) + 2*abs(control_sig_2)
                       + abs(self.state[6]) )
```

## Termination Condition
The episode was terminated according to the following condition:
```python
done = (self.t > self.t_max) or (abs(self.state[2]) > 0.75) or (abs(self.state[6]) > 0.75)
```

## OU Noise Process Parameters
The OU noise parameters were set as follows:
```
mu=0.0
theta=0.15
sigma=0.2
```

## Action Signal Scaling
The action signals were clipped as follows:
```python
control_sig_1, control_sig_2 = 0.1*action[0], 0.1*action[1]
```

## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion