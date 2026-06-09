import torch
from transformers import AutoTokenizer

from models.registry import MODEL_REGISTRY

class InferenceEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = MODEL_REGISTRY[model_name].from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
        MODEL_REGISTRY[model_name].configure_tokenizer(self.tokenizer)
        self.device = next(self.model.parameters()).device
        self.kv_cache = MODEL_REGISTRY[self.model_name].setupKVCache(self.model_name, self.device)

    def sample_token(self, logits: torch.Tensor, sampling_technique: str) -> int:
        if sampling_technique == 'greedy':
            return torch.argmax(logits, dim=-1)
        else:
            raise ValueError(f"{sampling_technique} not supported as of now! Raise a PR for adding support")

    def generate(self, prompts: list[str], max_new_tokens: int, sampling_technique: str='greedy') -> list[str]:

        with torch.no_grad():
            tokenized_prompts = self.tokenizer(prompts, padding=True, return_tensors='pt')["input_ids"].to(self.device)
            finished = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)
            token_count = torch.zeros(len(prompts), dtype=torch.int, device=self.device)

            prefill = True
            while not finished.all():
                if prefill:     # Prefill
                    logits = self.model(tokenized_prompts, self.kv_cache)[:, -1, :]
                    prefill = False
                else:           # Decode
                    logits = self.model(tokenized_prompts[:, -1:], self.kv_cache)[:, -1, :]

                new_encoded_tokens = self.sample_token(logits, sampling_technique)

                new_encoded_tokens[finished] = self.tokenizer.pad_token_id
                new_encoded_tokens = new_encoded_tokens.unsqueeze(1)

                tokenized_prompts = torch.cat([tokenized_prompts, new_encoded_tokens], dim=1)

                token_count += 1 
                finished = finished | (new_encoded_tokens.squeeze(1) == self.tokenizer.eos_token_id) | (token_count == max_new_tokens)
            
            self.kv_cache.reset()

        return self.tokenizer.batch_decode(tokenized_prompts, skip_special_tokens=True)