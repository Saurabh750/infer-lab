from transformers import GPT2Model as HF_GPT2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .model import GPT2LMHead

def load_weights_from_hf(model: 'GPT2LMHead') -> 'GPT2LMHead':
    hf_model = HF_GPT2.from_pretrained(model.model_name)
    sd = hf_model.state_dict()

    model.gpt2model.gpt2emb.token_embd.weight.data.copy_(sd['wte.weight'])
    model.gpt2model.gpt2emb.pos_embd.weight.data.copy_(sd['wpe.weight'])
    model.gpt2model.ln.weight.data.copy_(sd['ln_f.weight'])
    model.gpt2model.ln.bias.data.copy_(sd['ln_f.bias'])

    for i in range(12):
      model.gpt2model.gpt2blocks[i].ln_1.weight.data.copy_(sd[f'h.{i}.ln_1.weight'])
      model.gpt2model.gpt2blocks[i].ln_1.bias.data.copy_(sd[f'h.{i}.ln_1.bias'])
      model.gpt2model.gpt2blocks[i].attn.c_attn.weight.data.copy_(sd[f'h.{i}.attn.c_attn.weight'].T)
      model.gpt2model.gpt2blocks[i].attn.c_attn.bias.data.copy_(sd[f'h.{i}.attn.c_attn.bias'])
      model.gpt2model.gpt2blocks[i].attn.c_proj.weight.data.copy_(sd[f'h.{i}.attn.c_proj.weight'].T)
      model.gpt2model.gpt2blocks[i].attn.c_proj.bias.data.copy_(sd[f'h.{i}.attn.c_proj.bias'])
      model.gpt2model.gpt2blocks[i].ln_2.weight.data.copy_(sd[f'h.{i}.ln_2.weight'])
      model.gpt2model.gpt2blocks[i].ln_2.bias.data.copy_(sd[f'h.{i}.ln_2.bias'])
      model.gpt2model.gpt2blocks[i].mlp.c_fc.weight.data.copy_(sd[f'h.{i}.mlp.c_fc.weight'].T)
      model.gpt2model.gpt2blocks[i].mlp.c_fc.bias.data.copy_(sd[f'h.{i}.mlp.c_fc.bias'])
      model.gpt2model.gpt2blocks[i].mlp.c_proj.weight.data.copy_(sd[f'h.{i}.mlp.c_proj.weight'].T)
      model.gpt2model.gpt2blocks[i].mlp.c_proj.bias.data.copy_(sd[f'h.{i}.mlp.c_proj.bias'])

    model.eval()
    return model