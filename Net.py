import torch.nn as nn
import torch.nn.functional as F

from Args import args
n_columns = args.n_columns

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_columns, 128)
        self.fc2 = nn.Linear(128, 196)
        self.fc3 = nn.Linear(196, 2)

    def forward(self, x):
        x = x.view(-1, n_columns)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
# Duplicate class that takes n_columns as a parameter during construction
class AnalysisNet(nn.Module):
    def __init__(self, column_count):
        super(AnalysisNet, self).__init__()
        self.column_count = column_count
        
        self.fc1 = nn.Linear(self.column_count, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, self.column_count)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

# Exports: Net, AnalysisNet