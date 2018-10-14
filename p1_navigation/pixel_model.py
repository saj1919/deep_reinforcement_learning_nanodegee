import torch
import torch.nn as nn
import torch.nn.functional as F

# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, num_input_chnl, action_size, seed, fc1_units=64, fc2_units=64):
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(num_input_chnl, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)

#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, num_input_chnl, action_size, seed, num_filters = [8,32], fc_layers=[32,64]):
        """Initialize parameters and build model.
        Params
        ======
            num_input_chnl (int): Number of input channels
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(num_input_chnl, num_filters[0], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv1bnorm = nn.BatchNorm2d(num_filters[0])
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv2bnorm = nn.BatchNorm2d(num_filters[1])
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(num_filters[1]*21*21, fc_layers[0])
        self.fc1bnorm = nn.BatchNorm1d(fc_layers[0])
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc2bnorm = nn.BatchNorm1d(fc_layers[1])
        self.fc2relu = nn.ReLU()
        
        self.fc3 = nn.Linear(fc_layers[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = self.conv1(state)
        state = self.conv1bnorm(state)
        state = self.conv1relu(state)
        state = self.conv1maxp(state)

        state = self.conv2(state)
        state = self.conv2bnorm(state)
        state = self.conv2relu(state)
        state = self.conv2maxp(state)

        state = state.reshape((-1,32*21*21)) #reshape the output of conv2 before feeding into fc1 layer

        state = self.fc1(state)
        state = self.fc1bnorm(state)
        state = self.fc1relu(state)

        state = self.fc2(state)
        state = self.fc2bnorm(state)
        state = self.fc2relu(state)

        state = self.fc3(state)

        return state

