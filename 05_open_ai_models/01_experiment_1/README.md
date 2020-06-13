<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 01
## Intent
Initial experimentation to understand if the environment, agent, and ddpg training algorithm has been implemented correctly.

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
    reward = -50*(abs(self.state[2]) + abs(self.state[6]))
else:
    reward = 0.2 - 10*( abs(self.state[2]) + abs(control_sig_1)
                      + abs(self.state[3]) + abs(control_sig_2)
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
The action signals were not scaled for this experiment.


## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion