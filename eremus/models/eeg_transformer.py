import numpy as np
import torch

import math
from typing import Tuple

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class Model(nn.Module):

    def __init__(self, args):
        
        
        super(Model, self).__init__()
        args_defaults=dict(
            num_classes = 4,
            num_channels = 32, 
            d_model = 32,
            nhead = 2, 
            d_hid = 2048,
            nlayers = 1,
            dropout = 0.5, 
            verbose = False
        )
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Linear(self.d_model, self.num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, num_classes]
        """
        #src = self.encoder(src) * math.sqrt(self.d_model)
        if self.verbose:
            print(src.size(), src_mask.size())
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        if self.verbose:
            print(output.size())
        output = self.decoder(output)
        if self.verbose:
            print(output.size())
        # select the first token
        output = output[0, :, :]
        if self.verbose:
            print(output.size())
        output = torch.sigmoid(output)#.squeeze(1)
        if self.verbose:
            print(output.size())
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)