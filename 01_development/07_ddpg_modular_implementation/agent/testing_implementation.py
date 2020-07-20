import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import Model
import Model_original

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model_1 = Model_original.Actor(7, 2, (400,300), F.relu, 2).to(device)

    test_model_2 = Model.Actor(7, 2, 2).to(device)

    test_model_3 = Model_original.Critic(7, 2, (400,300), F.relu, 2).to(device)

    test_model_4 = Model.Critic(7, 2, 2).to(device)

    state = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    state = torch.from_numpy(state).float().to(device)

    action = np.array((1.0, 1.0))
    action = torch.from_numpy(action).float().to(device)
    
    print('\nNewly developed Actor model')
    print(state)
    print(test_model_1)
    print(test_model_1(state))
    
    print('\nOld Actor model')
    print(state)
    print(test_model_2)
    print(test_model_2(state))
    
    print('\nNewly developed Critic model')
    print(state)
    print(test_model_3)
    print(test_model_3(state, action))
    
    print('\nOld Critic model')
    print(state)
    print(test_model_4)
    print(test_model_4(state, action))
