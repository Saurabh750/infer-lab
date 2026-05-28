import torch
from transformers import GPT2Tokenizer

from models.gpt2.model import GPT2LMHead, GPT2Config
from models.gpt2.loader import load_weights_from_hf

def tokenize(prompt: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0)
    
def greedy_sampling(logits: torch.Tensor) -> int:
    return torch.argmax(logits, dim=-1)

def generate(model, prompt: str, tokenizer, max_new_tokens: int) -> str:

    with torch.no_grad():
        new_tokens = 0
        tokenized_prompt = tokenize(prompt, tokenizer)
        while new_tokens < max_new_tokens:
            logits = model(tokenized_prompt)[:, -1, :]
            new_encoded_token = greedy_sampling(logits)
            new_decoded_token = tokenizer.decode(new_encoded_token)
            prompt += new_decoded_token
            tokenized_prompt = tokenize(prompt, tokenizer)
            
            new_tokens += 1

    return prompt

def main():
    prompt = 'Who is President of France?'

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = load_weights_from_hf(GPT2LMHead(GPT2Config()))

    print(generate(model, prompt, tokenizer, max_new_tokens=5))

if __name__ == "__main__":
    main()
