import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, input_ids):
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name) -> 'BaseModel': 
        pass

    @classmethod
    def configure_tokenizer(cls, tokenizer):
        pass