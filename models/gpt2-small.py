from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class GPT2Config:
    vocab_size: int=50257
    max_seq_length: int=1024
    embedding_dim: int=768
    n_layers: int=12
    n_att_heads: int=12
    dropout: float=0.0

class GPT2Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embd = nn.Embedding(config.max_seq_length, config.embedding_dim)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos_indices = torch.arange(T, device=token_ids.device)
        return self.token_embd(token_ids) + self.pos_embd(pos_indices)