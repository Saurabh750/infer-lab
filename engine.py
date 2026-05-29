import torch
from transformers import AutoTokenizer

from models.registry import MODEL_REGISTRY

class InferenceEngine:
    def __init__(self, model_name: str):
        self.model = MODEL_REGISTRY[model_name].from_pretrained()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def sample_token(self, logits: torch.Tensor, sampling_technique: str) -> int:
        if sampling_technique == 'greedy':
            return torch.argmax(logits, dim=-1)
        else:
            raise ValueError(f"{sampling_technique} not supported as of now! Raise a PR for adding support")

    def generate(self, prompt: str, max_new_tokens: int, sampling_technique: str='greedy') -> str:

        with torch.no_grad():
            new_tokens = 0
            tokenized_prompt = torch.tensor(self.tokenizer(prompt)["input_ids"]).unsqueeze(0)

            while new_tokens < max_new_tokens:
                logits = self.model(tokenized_prompt)[:, -1, :]
                new_encoded_token = self.sample_token(logits, sampling_technique)
                new_decoded_token = self.tokenizer.decode(new_encoded_token)
                prompt += new_decoded_token
                tokenized_prompt = torch.tensor(self.tokenizer(prompt)["input_ids"]).unsqueeze(0)

                new_tokens += 1

        return prompt