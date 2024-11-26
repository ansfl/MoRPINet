import torch
from torch import nn


class DnetModel(nn.Module):

    def __init__(self, input_channels=6, output_features=1):

        super(DnetModel, self).__init__()

        self.input_channels = input_channels

        self.conv1 = nn.Conv1d(self.input_channels, 7, kernel_size=2, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.dp_c = nn.Dropout(0.1)

        self.fc1 = nn.Linear(305, 512)
        self.dp1 = nn.Dropout(0.5)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 32)
        self.dp2 = nn.Dropout(0.5)
        self.ln2 = nn.LayerNorm(32)

        self.fc3 = nn.Linear(32, output_features)

    def forward(self, x):
        c = self.relu(self.conv1(x))
        c = c.view(c.size(0), -1)
        c = self.dp_c(c)
        x = torch.cat((c, x.flatten(1)), 1)

        x = self.fc1(x)
        x = self.dp1(x)
        x = self.ln1(x)
        x = self.relu(x)

        x = self.relu(self.ln2(self.dp2(self.fc2(x))))
        x = self.fc3(x)
        return x
