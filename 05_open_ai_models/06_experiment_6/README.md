<img src="https://fundraising.blackbaud.com.au/wp-content/uploads/2016/08/CDU-LOGO-RGB-LHS-1200x628.jpg" alt="Charles Darwin University - ENG720 Honours Thesis" width="200" />

# Experiment 06
## Intent
Experiment 5 produced promising results, however, was unable to reduce the frequency to acceptable levels. This experiment adjusts the reward function by changing the scalar multipliers on the frequency and tieline penalties. The idea is that by increasing the penalty on the frequency deviations, the ddpg algorithm will have greater incentive to drive the frequency deviations to zero. The control action penalty has been left as is so that agent puts less priority on control action minimisation, however, it still has incentive to reduce controller action where possible.


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
reward =  - ( 2*abs(self.state[2])
            + 2*abs(self.state[3])
            + 2*abs(self.state[6])
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