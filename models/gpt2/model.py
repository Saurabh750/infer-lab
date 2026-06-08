import torch
import torch.nn as nn
import math
from dataclasses import dataclass

from models.base import BaseModel
from models.kvcache import StandardKVCache
from ..registry import register_model
from .loader import load_weights_from_hf

@dataclass
class GPT2Config:
    embedding_dim: int
    n_layers: int
    n_att_heads: int
    vocab_size: int=50257
    max_seq_length: int=1024
    dropout: float=0.0

    @classmethod
    def gpt2_small(cls):
        return cls(embedding_dim=768, n_layers=12, n_att_heads=12)

    @classmethod
    def gpt2_medium(cls):
        return cls(embedding_dim=1024, n_layers=24, n_att_heads=16)

    @classmethod
    def gpt2_large(cls):
        return cls(embedding_dim=1280, n_layers=36, n_att_heads=20)

    @classmethod
    def gpt2_xl(cls):
        return cls(embedding_dim=1600, n_layers=48, n_att_heads=25)

class GPT2Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embd = nn.Embedding(config.max_seq_length, config.embedding_dim)

    def forward(self, token_ids, prompt_length: int): # returns shape (B,T,C)
        B, T = token_ids.shape
        pos_indices = torch.arange(prompt_length + T, device=token_ids.device)
        return self.token_embd(token_ids) + self.pos_embd(pos_indices)
    
class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.n_att_heads = config.n_att_heads
        self.head_dim = self.embedding_dim // self.n_att_heads
        self.c_attn = nn.Linear(self.embedding_dim, 3*self.embedding_dim)
        self.c_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor, kv_cache: 'StandardKVCache', layer_idx: int): 
        # Input: (B,T,C)   # Output: 
        # B: Batch-size, T: Sequence Length, C: Embedding dim
        B, T, C = x.shape
        projection = self.c_attn(x)
        q, k, v = torch.split(projection, self.embedding_dim, dim=2)
        q_t = q.view(B, T, self.n_att_heads, self.head_dim)
        k_t = k.view(B, T, self.n_att_heads, self.head_dim)
        v_t = v.view(B, T, self.n_att_heads, self.head_dim)
        # q_t,k_t,v_t = (B,T,n_heads, head_dim)
        q = q_t.transpose(1,2)
        k = k_t.transpose(1,2)
        v = v_t.transpose(1,2)
        # q_t,k_t,v_t = (B,n_heads, T, head_dim)

        kv_cache.write(layer_idx, k, v)
        k_cache, v_cache = kv_cache.read(layer_idx)

        temp = q @ k_cache.transpose(2,3)  # (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T)
        temp = temp / math.sqrt(self.head_dim)
        if q.size(2) != 1:
            causal_mask = torch.triu(torch.full((T, T), 1, device=x.device), diagonal=1).bool()
            temp = temp.masked_fill(causal_mask, float('-inf'))
        
        temp = torch.nn.functional.softmax(temp, dim=-1)
        self_att = temp @ v_cache
        return self.c_proj(self_att.transpose(1,2).contiguous().view(B,T,C))
    
class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
        self.c_proj = nn.Linear(4*config.embedding_dim, config.embedding_dim)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = GPT2MLP(config)

    def forward(self, x, kv_cache: 'StandardKVCache', layer_idx: int):
        after_attn = x + self.attn(self.ln_1(x), kv_cache, layer_idx)
        return after_attn + self.mlp(self.ln_2(after_attn))

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_blocks = config.n_layers
        self.gpt2emb = GPT2Embeddings(config)
        self.gpt2blocks = nn.ModuleList([ GPT2Block(config) for _ in range(config.n_layers) ])
        self.ln = nn.LayerNorm(config.embedding_dim)

    def forward(self, x, kv_cache: 'StandardKVCache', prompt_length: int):
        x = self.gpt2emb(x, prompt_length)
        layer_idx = 0
        for block in self.gpt2blocks:
            x = block(x, kv_cache, layer_idx)
            layer_idx += 1
        x = self.ln(x)
        return x

@register_model
class GPT2LMHead(BaseModel):
    model_names = [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl"
    ]

    model_configs = {
        "openai-community/gpt2": GPT2Config.gpt2_small,
        "openai-community/gpt2-medium": GPT2Config.gpt2_medium,
        "openai-community/gpt2-large": GPT2Config.gpt2_large,
        "openai-community/gpt2-xl": GPT2Config.gpt2_xl
    }

    def __init__(self, config):
        super().__init__()
        self.gpt2model = GPT2Model(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.gpt2model.gpt2emb.token_embd.weight
        
    def forward(self, x, kv_cache: 'StandardKVCache'):
        prompt_length = 0 if kv_cache.k_cache[0] is None else kv_cache.k_cache[0].size(2)
        
        x = self.gpt2model(x, kv_cache, prompt_length)
        x = self.lm_head(x)
        return x
    
    @classmethod
    def from_pretrained(cls, model_name) -> 'GPT2LMHead':
        model_config = cls.model_configs[model_name]()
        model = cls(model_config)
        return load_weights_from_hf(model_name, model)
    
    @classmethod
    def configure_tokenizer(cls, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def setupKVCache(cls, model_name, device):
        model_config = cls.model_configs[model_name]()
        return StandardKVCache(model_config.n_layers, model_config.n_att_heads, model_config.embedding_dim // model_config.n_att_heads, device)