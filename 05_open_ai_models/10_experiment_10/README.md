<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 10
## Intent
Reducing the control action multiplier further in experiment 9 did not improve results.

Part of the ddpg algorithm is to overlay the control signals from the agent with an OU noise process, to help the agent explore the state-action space and reveal hidden optimal policies. This experiment will reduce the size of the stochastic noise that is currently being added to the control signal. This was motivated by a visual review of agent control signals from all previous experiments, which showed that the stochastic part of the control signal seemed to be large and whilst it encourages the agent to undertaken lots of exploration, might be causing the agent to converge to sub-optimal policies in the later stages of training.


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
theta=0.015
sigma=0.02
```

## Action Signal Scaling
```python
control_sig_1, control_sig_2 = 0.01*action[0], 0.01*action[1]
```

## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion