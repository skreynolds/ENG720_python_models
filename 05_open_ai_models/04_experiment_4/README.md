<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 04
## Intent
Experiment 3 was not successful in arriving at a policy that could control frequency. The agent still displayed a tendency to let the frequency extend above the 0.75 tripping boundary.

The reward function was was modified to give the agent a larger penalty for tripping the system, and a greater reward for every time step that the agent


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
            reward = -4200*(abs(self.state[2]) + abs(self.state[6]))
        else:
            reward = 2.5 - ( abs(self.state[2])
                           + abs(self.state[3])
                           + abs(self.state[6])
                           + abs(control_sig_1)
                           + abs(control_sig_2) )

return np.array(self.state), reward, done, {}
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