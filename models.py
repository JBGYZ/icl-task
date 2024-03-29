import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np

class Ridge(nn.Module):
    """A ridge linear regressor."""
    def __init__(self, lam):
        super(Ridge, self).__init__()
        self.lam = lam

    def forward(self, data, targets):
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # batch_size x n_points x 1
        preds = [torch.zeros(batch_size, dtype=data.dtype)]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1]) for _i in range(1, n_points)]
        )
        preds = torch.stack(preds, dim=1)
        return preds

    def predict(self, X, Y, test_x):
        _, _, n_dims = X.shape
        XT = X.transpose(1, 2)  # batch_size x n_dims x i
        XT_Y = torch.matmul(XT, Y)  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = torch.matmul(XT, X) + self.lam * torch.eye(n_dims, dtype=X.dtype)  # batch_size x n_dims x n_dims
        ws = torch.linalg.solve(ridge_matrix, XT_Y)  # batch_size x n_dims x 1
        # print(f"ws.shape: {ws.shape}")
        # print(f"test_x.shape: {test_x.shape}")
        
        pred = torch.matmul(test_x.unsqueeze(1), ws)  # @ should be ok (batched row times column)
        # print(pred)
        # return pred[:, 0, 0]
        return pred.squeeze()

  
class MLP(nn.Module):
    """A simple MLP with dropout but without batch normalization"""
    def __init__(self, input_dim, n_hidden_neurons, output_dim=30 ,n_hidden_layers=2, dropout=0.1):
        super(MLP, self).__init__()

        self.first_layer = nn.Linear(input_dim, n_hidden_neurons)
        # self.layers = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(n_hidden_neurons), nn.Linear(n_hidden_neurons, n_hidden_neurons)) for _ in range(n_hidden_layers)])        
        self.layers = nn.ModuleList([nn.Linear(n_hidden_neurons, n_hidden_neurons) for _ in range(n_hidden_layers)])        
        self.last_layer = nn.Linear(n_hidden_neurons, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x = x.reshape(x.size(0), -1) # flatten
        x = self.first_layer(x)
        for layer in self.layers:
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = layer(x)
        x = nn.functional.relu(x)
        x = self.last_layer(x)
        return x
    
class MLPMixer(nn.Module):
    """A simple MLP Mixer without batch normalization and dropout"""
    def __init__(self, input_dim, n_hidden_neurons, n_seq=16, output_dim=1 ,n_hidden_layers=2, dropout=0.1):
        super(MLPMixer, self).__init__()
        self.first_layer = nn.Linear(input_dim, n_hidden_neurons)
        self.second_layer = nn.Linear(n_seq, n_hidden_neurons)

        self.layers = nn.ModuleList([nn.Linear(n_hidden_neurons, n_hidden_neurons) for _ in range(n_hidden_layers)])     
          
        self.last_layer = nn.Linear(n_hidden_neurons, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x = self.first_layer(x)
        x = nn.functional.relu(x)
        x = x.transpose(1, 2)

        x = self.second_layer(x)
        x = nn.functional.relu(x)
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
            x = x.transpose(1, 2) # batch_size x n_seq x n_hidden_neurons

        x = self.last_layer(x[:, -1, :])
        return x
    
class MLPMixerFCN(nn.Module):
    """A simple one layer MLP Mixer plus deep FCN without batch normalization and dropout"""
    def __init__(self, input_dim, n_hidden_neurons, n_seq=16, output_dim=1 ,n_hidden_layers=2, dropout=0.1):
        super(MLPMixerFCN, self).__init__()
        self.first_layer = nn.Linear(input_dim, n_hidden_neurons)
        self.second_layer = nn.Linear(n_seq, n_hidden_neurons)

        self.reducer_layer = nn.Linear(n_hidden_neurons*n_hidden_neurons, n_hidden_neurons)
        self.layers = nn.ModuleList([nn.Linear(n_hidden_neurons, n_hidden_neurons) for _ in range(n_hidden_layers)])     
          
        self.last_layer = nn.Linear(n_hidden_neurons, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x = self.first_layer(x)
        x = nn.functional.relu(x)
        x = x.transpose(1, 2)

        x = self.second_layer(x)
        x = nn.functional.relu(x)
        x = x.transpose(1, 2)

        x = x.reshape(x.size(0), -1) # flatten
        x = self.reducer_layer(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
            # x = x.transpose(1, 2) # batch_size x n_seq x n_hidden_neurons

        x = self.last_layer(x)
        return x

class CNN(nn.Module):
    """An 2D Convulational Neural Network adapted for ICL linear regression."""
    def __init__(self,
                 embed_dim=8*20*5,
                 num_filters=20*5,
                 n_hidden_neurons=100,
                 n_hidden_layers=3,
                 n_dim = 4,
                 dropout=0.1):
        """
        The constructor for CNN class.

        Embed_dim and n_dim are provided in the main file.

        Args:
            num_filters (int): Nb of filters in the conv layer
            n_hidden_neurons (int): Nb of neurons in the FCN layers
            n_hidden_layers (int): Nb of FCN layers
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        # Conv layer 
        self.conv_layer = nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(1, n_dim+1),
                      stride = 1)
        
        self.first_layer = nn.Linear(embed_dim*2, n_hidden_neurons)
        # Fc layers
        self.layers = nn.ModuleList([nn.Linear(n_hidden_neurons, n_hidden_neurons) for _ in range(n_hidden_layers)])  
        
        self.last_layer = nn.Linear(n_hidden_neurons, 1)


    def forward(self, x: torch.Tensor):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape
                (batch_size, n_seq, n_dim)

        Returns:
            
        """
        x = x.unsqueeze(1) # batch_size x 1 x n_seq x n_dim
        x = self.conv_layer(x)
        # scalar product between x[1] items and its neighbors
        # x = (x[:, 1::2, :] * x[:,  ::2, :])

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(x)
        x = self.first_layer(x)
        for layer in self.layers:
            x = nn.functional.relu(x)
            x = layer(x)

        x = nn.functional.relu(x)
        x = self.last_layer(x)
        return x
class ScaleupEmbedding(nn.Module):
    """
    Learnable embedding from seq_len x input_dim to (seq_len/patch_size) x out_dim
    """
    def __init__(
        self, input_dim, out_dim, patch_size
    ):
        super().__init__() # input shape is (batch_size, seq_len, input_dim)
        self.patch_size = patch_size
        self.e = nn.Parameter( torch.randn(out_dim, input_dim, patch_size))

    def forward(self, x):
        return F.conv1d(x.transpose(1,2), self.e, bias=None, stride=self.patch_size).transpose(1,2)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, device="cuda"):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.freq = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).to(device)

    def forward(self, x, positions=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional positional indices. If not provided, positions will be generated.
        Returns:
            Tensor with rotary position embeddings added.
        """
        if positions is None:
            positions = torch.arange(0, x.size(1)).float().to(x.device)

        sin_embedding = torch.sin(positions.unsqueeze(-1) * self.freq)
        cos_embedding = torch.cos(positions.unsqueeze(-1) * self.freq)

        sin_cos_embedding = torch.cat([sin_embedding, cos_embedding], dim=-1)

        # Expand the embeddings to match the input shape
        sin_cos_embedding = sin_cos_embedding.unsqueeze(0).expand(x.size(0), -1, -1)

        # Apply the rotational transformation
        embeddings_rotated = x * sin_cos_embedding

        return embeddings_rotated

class PositionalEncoding(nn.Module):
    """
        Absolute positional encoding for short sequences.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(3000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class LearnedPositionalEncoding(nn.Module):
    """
        learned positional encoding for short sequences.
    """
    def __init__(self, d_model, max_seq_len=250):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(1), seq_len)
        position_embeddings = self.position_embeddings(positions).permute(1, 0, 2)
        x = x + position_embeddings
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention layer 
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        
        # Position-wise feedforward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization for both attention and feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # Multi-head self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Position-wise feedforward
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    """
        Transformer encoder module for classification. Two permutations in forward method
    """
    def __init__(self, d_model, num_layers, nhead, dim_feedforward, scaleup_dim, embedding_type, pos_encoder_type, dropout=0.1, ):
        super(TransformerEncoder, self).__init__()
        if embedding_type == "scaleup":
            self.embedding = ScaleupEmbedding(d_model, scaleup_dim, 1)
            d_model = scaleup_dim
        elif embedding_type == "none":
            self.embedding = nn.Identity()
        else:
            raise NameError("Specify a valid embedding type in [scaleup]")
        
        if pos_encoder_type == "absolute":
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        elif pos_encoder_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model)
        else:  
            raise NameError("Specify a valid positional encoder type in [absolute, learned]")
        # Stack multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 1)



    def forward(self, src, src_mask=None):
        # src = src.permute(0,2,1)
        src = self.embedding(src)
        # src = src.permute(0,2,1)
        src = src.permute(1,0,2)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        src = src.permute(1,0,2) # (batch_size, seq_len, embedding_dim)
        # src = src.reshape(src.shape[0], -1)
        src = self.classifier(src)
        return src[:, -1, :] # return only the last token's output
        # return src

class TransformerEncoderFCN(nn.Module):
    """
        One layer transformer encoder plus deep FCN
    """
    def __init__(self, d_model, num_layers, nhead, dim_feedforward, scaleup_dim, embedding_type, pos_encoder_type, dropout=0.1, ):
        super(TransformerEncoderFCN, self).__init__()
        if embedding_type == "scaleup":
            self.embedding = ScaleupEmbedding(d_model, scaleup_dim, 1)
            d_model = scaleup_dim
        elif embedding_type == "none":
            self.embedding = nn.Identity()
        else:
            raise NameError("Specify a valid embedding type in [scaleup]")
        
        if pos_encoder_type == "absolute":
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        elif pos_encoder_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model)
        else:  
            raise NameError("Specify a valid positional encoder type in [absolute, learned]")
        # Stack multiple encoder layers

        self.transformer_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.first_layer = nn.Linear(d_model*16, d_model)
        # self.layers = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(n_hidden_neurons), nn.Linear(n_hidden_neurons, n_hidden_neurons)) for _ in range(n_hidden_layers)])        
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = src.permute(1,0,2)
        src = self.pos_encoder(src)
        src = self.transformer_layer(src, src_mask)
        src = src.permute(1,0,2) # (batch_size, seq_len, embedding_dim)

        src = src.reshape(src.size(0), -1) # flatten
        src = self.first_layer(src)
        for layer in self.layers:
            src = layer(src)
            src = nn.functional.relu(src)
            src = self.dropout(src)
        # src = src.reshape(src.shape[0], -1)
        src = self.classifier(src)
        return src # return only the last token's output
        # return src


class StackedRNN(nn.Module):
    """A stacked RNN model."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StackedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the stacked RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through stacked RNN layers
        out, _ = self.rnn(x, h0)

        # Index hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# class CNNdeep(nn.Module):
#     """An 1D Convulational Neural Network."""
#     def __init__(self,
#                  embed_dim=8*20*5,
#                  num_filters=20*5,
#                  n_hidden_neurons=100,
#                  n_hidden_layers=3,
#                  n_dim = 4,
#                  dropout=0.1):
#         """
#         The constructor for CNN class.

#         Args:
#             vocab_size (int): Need to be specified when not pretrained word
#                 embeddings are not used.
#             embed_dim (int): Dimension of word vectors. Need to be specified
#                 when pretrained word embeddings are not used. Default: 300
#             filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
#             num_filters (List[int]): List of number of filters, has the same
#                 length as `filter_sizes`. Default: [100, 100, 100]
#             n_classes (int): Number of classes. Default: 2
#             dropout (float): Dropout rate. Default: 0.5
#         """

#         super(CNNdeep, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_filters = num_filters
#         # Conv layer 
#         self.conv_layer = nn.Conv2d(in_channels=1,
#                       out_channels=num_filters,
#                       kernel_size=(3,3),
#                       stride = 1,
#                       padding=1)
        
#         self.layers_cnn = nn.ModuleList([nn.Conv2d(in_channels=num_filters,
#                       out_channels=num_filters,
#                       kernel_size=(3,3),
#                       stride = 1,
#                       padding=1) for _ in range(2)])
        
#         # self.first_layer = nn.Linear(embed_dim, n_hidden_neurons)
#         self.first_layer = nn.Linear(16*5*num_filters, n_hidden_neurons)

#         # Fc layers
#         self.layers = nn.ModuleList([nn.Linear(n_hidden_neurons, n_hidden_neurons) for _ in range(2)])  
        
#         self.last_layer = nn.Linear(n_hidden_neurons, 1)


#     def forward(self, x: torch.Tensor):
#         """Perform a forward pass through the network.

#         Args:
#             x (torch.Tensor): A tensor of token ids with shape
#                 (batch_size, n_seq, n_dim)

#         Returns:
            
#         """
#         x = x.unsqueeze(1) # batch_size x 1 x n_seq x n_dim
#         x = self.conv_layer(x) # batch_size x num_filters x n_seq x 1
#         # scalar product between x[1] items and its neighbors
#         # x = (x[:, 1::2, :] * x[:,  ::2, :])
#         x = x.squeeze(-1) # batch_size x num_filters x n_seq
#         for layer in self.layers_cnn:
#             x = layer(x)
#             x = nn.functional.relu(x)            
#             # print(x.shape)
#         x = x.view(x.size(0), -1)
#         x = nn.functional.relu(x)
#         x = self.first_layer(x)
#         for layer in self.layers:
#             x = nn.functional.relu(x)
#             x = layer(x)

#         x = nn.functional.relu(x)
#         x = self.last_layer(x)
#         return x

