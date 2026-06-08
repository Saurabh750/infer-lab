import torch
from abc import ABC, abstractmethod

class KVCache(ABC):
    @abstractmethod
    def read(self, layer_idx):
        pass

    @abstractmethod
    def write(self, layer_idx, K, V):
        pass

    @abstractmethod
    def reset(self):
        pass

class StandardKVCache(KVCache):
    def __init__(self, n_layers: int, n_heads: int, head_dim: int, device: str):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        self.k_cache, self.v_cache = [None]*self.n_layers, [None]*self.n_layers

    def read(self, layer_idx: int):
        
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def write(self, layer_idx: int, K: torch.Tensor, V: torch.Tensor):
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = K
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], K], dim=2)
        
        if self.v_cache[layer_idx] is None:
            self.v_cache[layer_idx] = V
        else:
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], V], dim=2)

    def reset(self):
        self.k_cache, self.v_cache = [None] * self.n_layers, [None] * self.n_layers