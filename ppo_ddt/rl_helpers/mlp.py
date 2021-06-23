"""
Created by Andrew Silva on 10/26/20
"""
import torch
import torch.nn as nn
import typing as t
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: t.List[int]):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.encoder = None

        self.sig = nn.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        last_dim = input_dim
        modules = []
        for h in hidden_layers:
            modules.append(nn.Linear(last_dim, h))
            torch.nn.init.xavier_uniform_(modules[-1].weight, gain=1)
            torch.nn.init.constant_(modules[-1].bias, 0)
            modules.append(nn.ReLU())
            last_dim = h

        self.encoder = nn.Sequential(*modules)
        self.action_head = nn.Linear(last_dim, output_dim)
        self.log_std_head = nn.Linear(last_dim, output_dim)
        self.apply(weight_init)

    def forward(self, state_data):
        state_data = self.sig(self.lin1(state_data))
        state_data = self.encoder(state_data)
        action = self.action_head(state_data)
        std = self.log_std_head(state_data)
        std = torch.clamp(std, min=-20, max=2)  # -20 log std min, 2 log std max
        std = torch.exp(std)
        return action, std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weight_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = torch.nn.functional.relu(self.linear1(xu))
        x1 = torch.nn.functional.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.nn.functional.relu(self.linear4(xu))
        x2 = torch.nn.functional.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class QMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: t.List[int]):
        super(QMLP, self).__init__()
        self.lin1_1 = nn.Linear(input_dim, input_dim)
        self.lin1_2 = nn.Linear(input_dim, input_dim)
        self.act = nn.ReLU()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        last_dim = input_dim
        modules = []
        for h in hidden_layers:
            modules.append(nn.Linear(last_dim, h))
            torch.nn.init.xavier_uniform_(modules[-1].weight, gain=1)
            torch.nn.init.constant_(modules[-1].bias, 0)
            modules.append(nn.ReLU())
            last_dim = h
        self.encoder_1 = nn.Sequential(*modules)
        self.encoder_2 = nn.Sequential(*modules)

        self.q_head_1 = nn.Linear(last_dim, output_dim)
        self.q_head_2 = nn.Linear(last_dim, output_dim)
        self.apply(weight_init)

    def forward(self, state_data, action_data=None):
        if action_data is not None:
            state_data = torch.cat((state_data, action_data), dim=1)
        q1_data = self.act(self.lin1_1(state_data))
        q2_data = self.act(self.lin1_2(state_data))
        q1_data = self.encoder_1(q1_data)
        q2_data = self.encoder_2(q2_data)
        q_1 = self.q_head_1(q1_data)
        q_2 = self.q_head_2(q2_data)
        return q_1, q_2
