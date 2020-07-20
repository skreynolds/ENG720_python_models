import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_units, gate, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        dims = (state_size,) + hidden_units + (action_size,)

        self.layers = [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])]

        for i, layer in enumerate(self.layers):
            self.add_module("fc"+str(i+1), layer)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return torch.tanh(self.layers[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_units, gate, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        dims = hidden_units + (1,)

        self.layers = [nn.Linear(state_size, dims[0]),
                       nn.Linear(dims[0]+action_size, dims[1])]

        for dim_in, dim_out in zip(dims[1:-1], dims[2:]):
            self.layers.append(nn.Linear(dim_in, dim_out))

        for i, layer in enumerate(self.layers):
            self.add_module("fc"+str(i+1), layer)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, action):
        """
        Build a critic (value) network that maps
           (state, action) pairs -> Q-values.
        """
        xs = self.gate(self.layers[0](x))
        x = torch.cat((xs, action), dim=1)
        for layer in self.layers[1:-1]:
            x = self.gate(layer(x))
        return self.layers[-1](x)