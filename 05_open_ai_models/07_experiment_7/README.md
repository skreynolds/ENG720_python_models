<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 07
## Intent
Experiment 6 demonstrated improved results over experiment 5, however, the agent is still unable to reduce the frequency to a similar level as a basic PI controller. This experiment will increase the scalar multiple on the frequency deviations ever further.


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
reward =  - ( 4*abs(self.state[2])
            + 2*abs(self.state[3])
            + 4*abs(self.state[6])
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
```python
control_sig_1, control_sig_2 = 0.1*action[0], 0.1*action[1]
```

## Cumulative Reward Over Time


## Best Agent Performance


## Discussion and Conclusion