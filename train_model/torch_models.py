import torch.nn as nn
import torch


# model type: MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=128):
        '''
        @param input_dim: dimensionality of input features
        @param output_dim: number of classes for prediction
        @param neurons: dimensionality of hidden units at ALL layers
        '''
        super(MLP, self).__init__()
        # The torch.nn.Sequential class is used to implement a simple Sequential connection model, a four-layer neural network
        self.linear = nn.Sequential(
            nn.Linear(input_dim, neurons),
            nn.Sigmoid(),
            nn.Linear(neurons, neurons),
            nn.Sigmoid(),
            nn.Linear(neurons, neurons),
            nn.Sigmoid(),
            nn.Linear(neurons, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x0 = self.linear(x)
        rej_NaSO4, rej_MgCl2, rej_NaCl, rej_PEG = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3]
        sel1 = torch.log(torch.clip(
            (1 - rej_NaCl.detach()) / (1 - rej_NaSO4), min=0.0001)).unsqueeze(dim=1)
        sel2 = torch.log(torch.clip(
            (1 - rej_NaCl.detach()) / (1 - rej_MgCl2), min=0.0001)).unsqueeze(dim=1)
        sel3 = torch.log(torch.clip(
            (1 - rej_NaCl.detach()) / (1 - rej_PEG), min=0.0001)).unsqueeze(dim=1)
        # print(x0.shape, sel1.shape)
        x0 = torch.cat([x0, sel1, sel2, sel3], dim=-1)
        return x0


# model type: MLP_skip
class MLP_skip(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=128):
        '''
        @param input_dim: dimensionality of input features
        @param output_dim: number of classes for prediction
        @param neurons: dimensionality of hidden units at ALL layers
        '''
        super(MLP_skip, self).__init__() # Call the parent class
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, neurons),
            nn.Sigmoid()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(neurons, neurons),
            nn.Sigmoid()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(neurons, neurons),
            nn.Sigmoid()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(neurons, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = (self.linear2(x1) + x1)
        x3 = (self.linear3(x2) + x2)
        x0 = self.linear4(x3)
        rej_NaSO4, rej_MgCl2, rej_NaCl, rej_PEG = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3]
        sel1 = torch.log(torch.clip(
            (1 - rej_NaCl) / (1 - rej_NaSO4), min=0.0001)).unsqueeze(dim=1)
        sel2 = torch.log(torch.clip(
            (1 - rej_NaCl) / (1 - rej_MgCl2), min=0.0001)).unsqueeze(dim=1)
        sel3 = torch.log(torch.clip(
            (1 - rej_NaCl) / (1 - rej_PEG), min=0.0001)).unsqueeze(dim=1)
        # print(x0.shape, sel1.shape)
        x0 = torch.cat([x0, sel1, sel2, sel3], dim=-1)
        return x0
