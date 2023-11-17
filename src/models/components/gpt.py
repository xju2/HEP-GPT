import torch
from torch import Tensor
import torch.nn as nn

import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from src.utils import utils

log = utils.get_pylogger(__name__)

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


class GPT(nn.Module):
    def __init__(self,
                 block_size: int,
                 vocab_size: int,
                 n_layer: int,
                 n_head: int,
                 n_embd: int,
                 dropout: float,
                 bias: bool,
                 dim_feedforward: int,
                 norm_first: bool,
                 activation: str,
                 padding_idx: int):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.dim_feedforward = dim_feedforward
        self.norm_first = norm_first
        self.activation = activation
        self.padding_idx = padding_idx

        self.model_type = "GPT"

        self.register_buffer(
            "src_mask",
            nn.Transformer.generate_square_subsequent_mask(self.block_size)
        )

        # token embeddings, [B, T] -> [B, T, D]
        # B: batch size
        # T: block size
        # D: embedding dimension
        self.tok_emb = nn.Embedding(
            self.vocab_size,
            self.n_embd,
            self.padding_idx)

        # positional embeddings, [B, T, D] -> [B, T, D]
        self.pos_emb = PositionEncoding(self.n_embd, self.dropout, self.block_size)

        self.drop = nn.Dropout(self.dropout)

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            self.n_embd,
            self.n_head,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            norm_first=self.norm_first,
            batch_first=True,
            bias=self.bias,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=self.n_layer)

        self.decoder = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.tok_emb.weight = (
            self.decoder.weight
        )  # https://paperswithcode.com/method/weight-tying

        log.info("number of parameters: {:.2f}M".format(self.get_num_params() / 1e6))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx: Tensor):
        """
        Arguments:
            idx: Tensor, shape ``[batch_size, seq_len]``
        """
        idx = self.tok_emb(idx) * math.sqrt(self.n_embd)
        idx = self.pos_emb(idx)

        embedding = self.encoder(idx, mask=self.src_mask, is_causal=True)
        logits = self.decoder(embedding)
        return logits
