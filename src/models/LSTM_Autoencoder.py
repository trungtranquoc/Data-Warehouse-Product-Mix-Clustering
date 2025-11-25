import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, padded_X: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            padded_X: (batch_size, seq_len, input_size)
            lengths: (batch_size) - Length of each sequence
        Returns:
            hidden_state: The final hidden state (embedding) of the last layer.
        """
        # Pack the sequence to ignore padding during encoding
        packed_X = rnn_utils.pack_padded_sequence(
            padded_X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        _, (h_n, _) = self.lstm(packed_X)
        
        # Extract the output of the last layer to serve as the embedding
        embedding = h_n[-1]                         # h_n shape: (num_layers, batch_size, hidden_size)
        embedding = self.layer_norm(embedding)

        return embedding

class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(Decoder, self).__init__()
        # Note: The input to the decoder LSTM is the embedding size (hidden_size of encoder)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, embedding: torch.Tensor, lengths: torch.Tensor, max_len: int):
        """
        Args:
            embedding: (batch_size, hidden_size) - The compressed representation
            lengths: (batch_size) - Original lengths to repack correctly
            max_len: int - The maximum length to reconstruct
        """
        # 1. Repeat Vector Strategy
        # We repeat the embedding 'max_len' times to create the input for the Decoder.
        # Input shape becomes: (batch_size, max_len, hidden_size)
        decoder_input = embedding.unsqueeze(1).repeat(1, max_len, 1)
        
        # 2. Pack the repeated input
        packed_input = rnn_utils.pack_padded_sequence(
            decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 3. Pass through Decoder LSTM
        # We do NOT pass the encoder hidden state here. We let the LSTM learn to reconstruct purely from the input sequence of embeddings.
        packed_out, _ = self.lstm(packed_input)
        
        # 4. Unpack
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)          # out shape: (batch_size, max_len, hidden_size)
        
        # 5. Project back to original feature space
        reconstruction = self.linear(out)
        
        return reconstruction

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size=hidden_size, hidden_size=hidden_size, output_size=input_size, num_layers=num_layers)
        
    def forward(self, padded_X: torch.Tensor, lengths: torch.Tensor):
        embedding = self.encoder(padded_X, lengths)

        max_len = padded_X.shape[1]                     # For the decoder to know how much to pad/reconstruct
        reconstructed = self.decoder(embedding, lengths, max_len)
        
        return reconstructed, embedding
    
    def encode(self, padded_X: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(padded_X, lengths)