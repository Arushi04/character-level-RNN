import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=50, hidden_dim=100):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input):
        #input_dim = batch*seq_len(length of name)

        embedded = self.embedding(input) #embedding-dim(input dimension) : batch * seq_len * embed_dim
        #B, L, D = embedded.size()
        #embedded = embedded.view(L, B, D) #[20, 5, 50]

        _, (h_n, c_n) = self.lstm(embedded)
        #print(h_n.shape) #[1, 5, 100]
        hidden = torch.squeeze(h_n)
        #linear_output = self.fc(self.relu(hidden))
        linear_output = self.fc(hidden)
        #seq_len * batch * input_size -> (seq_len, batch, hidden_size), h_n, c_n
        return linear_output


def main():
    model = RNN(20, 10)
    #print(model)
    input = torch.randint(19, (5, 20))
    model(input)


if __name__ == '__main__':
    main()
