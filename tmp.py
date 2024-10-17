import torch
from torch.nn import Module
import torch.nn as nn

class LSTMSelfAttn(Module):
    def __init__(self, vocab_size, hidden_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.attn_w = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedding = self.embed(x)
        lstm_output = self.lstm(embedding)[0]
        attn = self.attn_w(lstm_output)
        attn_score = torch.softmax(attn.squeeze(), -1)

        output = (lstm_output * attn_score.unsqueeze(-1)).sum(1)

        output
        print(output.shape)
        


if __name__ == '__main__':
    model = LSTMSelfAttn(100, 256)
    x = torch.randint(low=0, high=100, size=(2, 10))
    model(x)
