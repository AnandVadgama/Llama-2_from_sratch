import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 #number of heads for the query
    n_kv_heads: Optional[int] = None # number of heads for the K and V
    vocab_size: int = -1 # this will be set when we load tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # as written in the paper the dim of the embedding must be even
    assert head_dim % 2 == 0, "the dim of the embedding must be even"

    # build the theta parameter
    # according to the formula theta_i = 10000.0 ^ (-2(i-1)/dim) for i = [1,2,3.....dim/2]
    # shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim /2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the position (the "m" parameter)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using outer product
    # shape: (seq_len) outerproduct * (head_dim/2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # we can compute complex numbers in polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x: torch.tensor, freqs_complex: torch.tensor, device: str):
    # in this we are pairing two consicutive number and pairing them to and the first one is real number and the second one is imagenary (i)
    # shape: (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # dot product between two complex numbers
    # (B, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # in this we are unpairing the real and imagenary (i*) numbers
    # (B, seq_len, h, head_dim / 2) -> (B, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # in here we are flatten the matrics
    # (B, seq_len, h, head_dim / 2, 2) -> (B, seq_len, h, head_dim)
    x_out = torch.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # the gamma parameter
        self.weights = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt: 1 / sqrt()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.tensor):
        # (dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return self.weights * self._norm(x.float).type_as(x)

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before the self attention
        self.attention_norm = RMSNorm(args.dim, eps= args.norm_eps)
        # normalization before the ffn
        self.ffn_norm = RMSNorm(args.dim, eps= args.norm_eps)

    def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.vocab_size != -1 , "vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def foward(self,tokens: torch.tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time processed"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pair (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # consicutively apply all the encoder layer
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
