import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.hh = nn.Linear(input_dim, hidden_dim)
        self.hx = nn.Linear(input_dim, hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.input_dim, self.hidden_dim)
        x = self.hx(x)
        h = self.hh(h)
        return torch.tanh(x + h)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        self.rnn_cell = RNNCell(input_dim, hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, h=None):
        """
        :param x: [n_len, batch_size, dim]
        :param h: [----------------------]
        :return: [batch_size, dim]
        """
        n_len, batch_size, dim = x.shape
        x_t = None
        h_t_1 = torch.zeros(batch_size, self.hidden_dim)
        for t in range(n_len):
            x_t = x[t]
            x_t = self.rnn_cell(x_t, h_t_1)
            h_t_1 = x_t
        return x_t


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.ix = nn.Linear(input_dim, hidden_dim)
        self.ih = nn.Linear(input_dim, hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(input_dim, hidden_dim)
        self.ox = nn.Linear(input_dim, hidden_dim)
        self.oh = nn.Linear(input_dim, hidden_dim)
        self.cx_ = nn.Linear(input_dim, hidden_dim)
        self.ch_ = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, h, c_1):
        i = torch.sigmoid(self.ix(x) + self.ih(h))
        f = torch.sigmoid(self.fx(x) + self.fh(h))
        o = torch.sigmoid(self.ox(x) + self.oh(h))
        c_ = torch.tanh(self.cx_(x) + self.ch_(h))
        c = torch.sigmoid(f * c_1 + i * c_)
        h = torch.tanh(c) * o
        return h, c


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm_cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, x, h=None, c=None):
        n_len, batch_size, embed_dim = x.shape
        if h is None:
            h_t_1 = torch.zeros(batch_size, embed_dim)
            c_t_1 = torch.zeros(batch_size, embed_dim)
        else:
            h_t_1 = h
            c_t_1 = c

        for t in range(n_len):
            x_t = x[t]
            x_t, c_t_1 = self.lstm_cell(x_t, h_t_1, c_t_1)
            h_t_1 = x_t
        return h_t_1


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.zx = nn.Linear(input_dim, hidden_dim)
        self.zh = nn.Linear(input_dim, hidden_dim)
        self.rx = nn.Linear(input_dim, hidden_dim)
        self.rh = nn.Linear(input_dim, hidden_dim)
        self.hx = nn.Linear(input_dim, hidden_dim)
        self.hh = nn.Linear(input_dim, hidden_dim)
        # self.cx_ = nn.Linear(input_dim, hidden_dim)
        # self.ch_ = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, h):
        z = torch.sigmoid(self.zx(x) + self.zh(h))
        r = torch.sigmoid(self.rx(x) + self.rh(h))
        h_ = torch.sigmoid(self.hx(x) + self.hh(r * h))
        h = (1 - z) * h + z * h_
        return h


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.lstm_cell = GRUCell(input_dim, hidden_dim)

    def forward(self, x, h=None, c=None):
        n_len, batch_size, embed_dim = x.shape
        if h is None:
            h_t_1 = torch.zeros(batch_size, embed_dim)
        else:
            h_t_1 = h

        for t in range(n_len):
            x_t = x[t]
            x_t = self.lstm_cell(x_t, h_t_1)
            h_t_1 = x_t
        return h_t_1
