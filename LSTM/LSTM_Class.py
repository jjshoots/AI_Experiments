import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super().__init__()
        self.net_layers = 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.drop_probability = 0.5

        self.lstm_L1 = nn.LSTM(self.input_dim, self.hidden_dim, self.net_layers, batch_first=True, dropout=self.drop_probability)
        self.linear_L2 = nn.Linear(self.hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(self.drop_probability)

    def forward(self, input, hidden):
        L1_output, hidden = self.lstm_L1(input, hidden)
        L1_dropped = self.dropout(L1_output)
        output = self.linear_L2(L1_dropped)
        return output, hidden

    def initHiddenLayer(self, test):
        if not test:
            hidden_state = torch.zeros(self.net_layers, self.batch_size, self.hidden_dim)
            cell_state = torch.zeros(self.net_layers, self.batch_size, self.hidden_dim)
            hidden = (hidden_state, cell_state)
            return hidden
        else:
            hidden_state = torch.zeros(self.net_layers, 1, self.hidden_dim)
            cell_state = torch.zeros(self.net_layers, 1, self.hidden_dim)
            hidden = (hidden_state, cell_state)
            return hidden
