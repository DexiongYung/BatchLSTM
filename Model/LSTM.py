import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, num_layers: int, hidden_sz: int, output_size: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_sz
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_sz, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_sz, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        output, hidden = self.lstm(input, hidden)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
