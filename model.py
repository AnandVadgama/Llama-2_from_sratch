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

    device:str = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # the gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt: 1 / sqrt()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.tensor):
        # (dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

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
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
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
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.tensor, n_rep: int)-> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim= x.shape
    if n_rep== 1:
        return x
    else:
        # (B, seq_len, n_kv_heads, head_dim)  -> (B, seq_len, n_kv_heads, n_rep, head_dim) -> (B, seq_len, n_kv_heads * n_rep, head_dim)
        return(
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # idicates the number of heads for the keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # indecates the number of heads for the queries 
        self.n_heads_q = args.n_heads
        # indecates how many times heads of keys and values should be repeated to match of the heads of query
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # indicates the dim of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, dim)

        # apply the wq, wk, wv matrices to the query, key and value
        # (B, 1, dim) --> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) --> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_KV * head_dim) --> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * head_dim) --> (B, 1, H_KV, head_dim)
        xk = xk.view(batch_size,seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size,seq_len, self.n_kv_heads, self.head_dim)

        # does not change the shape of the tensor
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xq, freqs_complex, device=x.device)

        # replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # retrieve all the key and values so far
        # (B, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:, 0:start_pos+seq_len]
        values = self.cache_v[:, 0:start_pos+seq_len]

        # repeat the heads of the K and V to reach the number of heads of the query
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, h_q, head_dim) -> (B, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, h_q, 1, head_dim) @ (B, h_q, head_dim, seq_len) -> (B, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len) @ (B, h_q, seq_len_kv, head_dim) -> (B, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, h_q, 1, head_dim) -> (B, 1, h_q, head_dim) -> (B, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden_dim dim to the nearest multiple of the multiple_of parameter
        hidden_dim = int(args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of))

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.tensor):
        swiss = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swiss * x_v
        x = self.w2(x)
        return x


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
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self,tokens: torch.tensor, start_pos: int):
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
