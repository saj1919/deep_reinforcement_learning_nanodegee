import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, final_gate=F.tanh):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.final_gate = final_gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):        
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, states):
        x = self.normalizer(states)
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.final_gate(self.output(x))

    
class Critic(nn.Module):
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, dropout=0.2):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList()
        count = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if count == 1:
                self.layers.append(nn.Linear(dim_in+action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            count += 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        xs = self.normalizer(states)
        xs = self.gate(self.layers[0](xs))
        x = torch.cat((xs, actions), dim=1)
        for i in range(1, len(self.layers)):
            x = self.gate(self.layers[i](x))
        x = self.dropout(x)
        return self.output(x)

