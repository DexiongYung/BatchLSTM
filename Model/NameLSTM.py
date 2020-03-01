import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class NameLSTM(nn.Module):
    def __init__(self, inputs: dict, outputs: dict, hidden_sz: int = 256, num_layers: int = 5, embed_dim: int = 3):
        super().__init__()

        self.inputs = inputs
        self.outputs = outputs

        self.num_layers = num_layers
        self.hidden_sz = hidden_sz
        self.embed_dim = embed_dim
        self.softmax = nn.Softmax(2)

        # don't count the padding tag for the classifier output
        self.nb_outputs = len(self.outputs) - 1

        # build embedding layer first
        nb_inputs_words = len(self.inputs)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.inputs['<PAD>']

        self.word_embedding = nn.Embedding(nb_inputs_words, embed_dim, padding_idx=padding_idx)

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_sz,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # output layer which projects back to output space
        self.hidden_to_output = nn.Linear(self.hidden_sz, self.nb_outputs)

    def forward(self, X: list, X_lens: list, hidden: torch.Tensor = None):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        batch_size, seq_len = X.size()

        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_sz)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        if hidden is None:
            X, self.hidden = self.lstm(X)
        else:
            X, self.hidden = self.lstm(X, hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, hidden_sz) -> (batch_size * seq_len, hidden_sz)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_output(X)

        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_sz) -> (batch_size, seq_len, nb_outputs)
        X = F.log_softmax(X, dim=1)

        # reshape (batch_size, seq_len, nb_outputs)
        Y_hat = self.softmax(X.view(batch_size, seq_len, self.nb_outputs))

        return Y_hat, self.hidden
