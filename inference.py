import torch


from models.gpt2small import GPT2LMHead, GPT2Config
from load_weights import load_weights_from_hf

def tokenize(prompt: str) -> torch.Tensor:
    pass

def greedy_sampling(logits: torch.Tensor) -> int:
    pass

def generate(model, prompt: str, max_new_tokens: int) -> str:
    new_tokens = 0
    tokenized_prompt = tokenize(prompt)
    while new_tokens < max_new_tokens:
        model(tokenized_prompt)


def main():
    prompt = 'Who is President of France?'

    model = load_weights_from_hf(GPT2LMHead(GPT2Config()))

    print(generate(model, prompt, max_new_tokens=1))