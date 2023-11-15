from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn import LayerNorm
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class PositionEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    Taken from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    dim_feedforward: int = 2048
    norm_first: bool = False
    activation: str = "gelu"
    padding_idx: int = None

class GPT(nn.Module):
    """Adapt from https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.model_type = "GPT"
        self.config = config

        # token embeddings, [B, T] -> [B, T, D]
        # B: batch size
        # T: block size
        # D: embedding dimension
        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd,
            config.padding_idx)

        # positional embeddings, [B, T, D] -> [B, T, D]
        self.pos_emb = PositionEncoding(config.n_embd, config.dropout, config.block_size)

        self.drop = nn.Dropout(config.dropout)

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            config.n_embd,
            config.n_head,
            config.dim_feedforward,
            config.dropout,
            config.activation,
            norm_first=config.norm_first,
            batch_first=True,
            bias=config.bias,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.tok_emb.weight = (
            self.decoder.weight
        )  # https://paperswithcode.com/method/weight-tying

        print("number of parameters: {:.2f}M".format(self.get_num_params() / 1e6))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx: Tensor, src_mask: Tensor = None):
        """
        Arguments:
            idx: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
        """
        seq_len = idx.size(1)
        idx = self.tok_emb(idx) * math.sqrt(self.config.n_embd)
        idx = self.pos_emb(idx)
        if src_mask is None:
            device = idx.device
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        embedding = self.encoder(idx, mask=src_mask, is_causal=True)
        logits = self.decoder(embedding)
        return logits
