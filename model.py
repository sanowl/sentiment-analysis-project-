import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(SentimentModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # Embed the input tokens
        embedded = self.embedding(input_ids)

        # Pack the embedded sequences to handle variable-length sequences efficiently
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths=attention_mask.sum(1), batch_first=True, enforce_sorted=False
        )

        # Pass the packed sequences through the LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack the output sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Get the hidden state of the last LSTM layer
        hidden = hidden[-1]

        # Pass the hidden state through the fully connected layer for sentiment prediction
        output = self.fc(hidden)

        return output
