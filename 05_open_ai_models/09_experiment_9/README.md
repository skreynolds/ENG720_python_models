<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 09
## Intent
Experiment 8 did not show material improvement over experiment 7. The agent seems to have trouble reducing the frequency to zero deviation. The agent in experiments 6, 7, and 8 seems to be able to reduce the frequency to a near zero value, however, there the agents seem to converge on a policy where the frequncy settles on a value 0.02Hz offset for the 0Hz level.

This experiment will reduce the control action multiplier further to provide the agent with the ability to use finer control adjustment. Note that reducing the control action in this way will most likely have a detrimental impact on the controllers ability to deal with power demand step changes other than the one seen in this experiment (i.e. the control action might be limited so that the controller does not have the necessary signal magnitude to correct a sufficiently large step change).


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
reward =  - ( 8*abs(self.state[2])
            + 4*abs(self.state[3])
            + 8*abs(self.state[6])
            + 10*abs(control_sig_1)
            + 10*abs(control_sig_2) )
```

## Termination Condition
The episode was terminated according to the following condition:
```python
done = (self.t > self.t_max)
```

## OU Noise Process Parameters
The OU noise parameters were set as follows:
```
mu=0.0
theta=0.15
sigma=0.2
```

## Action Signal Scaling
```python
control_sig_1, control_sig_2 = 0.01*action[0], 0.01*action[1]
```

## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion