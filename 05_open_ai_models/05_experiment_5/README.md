<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 05
## Intent
Experiment 4 was not successful in controlling the frequency. A visual analysis of the agent's performance in this experiment highlighted that the boundary tripping, and large negative reward may be causing unintended behavior to manifest in the policy, leading to suboptimal behavior.

This experiment has removed the termination condition when the agent moves 0.75 away from the reference frequency. The agent now has the entire state space to explore. With the removal of this termination condition, the agent does not need to be rewarded for each step that it takes and instead a simpler reward function can be implemented where the agent only receives a penalty at each time step based on the distance from the reference frequency, the tie-line power deviation, and the control action taken.

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
reward =  - ( abs(self.state[2])
            + abs(self.state[3])
            + abs(self.state[6])
            + abs(control_sig_1)
            + abs(control_sig_2) )
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
The action signals were not scaled for this experiment.


## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion