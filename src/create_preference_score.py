import sys
sys.path.append('/fl-ift/med/hujunchao/git_root/llama-recipes-main/src/llama_recipes')
from llama_recipes.utils.dpo_dataset import DPODataset, get_collate_fn
from llama_recipes.qwen_dpo.modeling_qwen_dpo import QWenLMHeadModelDPO
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from llama_recipes.qwen_dpo.dpo_utils import preference_loss, _get_batch_logps, concatenated_inputs
import torch
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_file, load_file
# reference_cache = load_file('reference_score2.safetensors')
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
model_path = '/fl-ift/med/hujunchao/models/dpo_14b_to_72b_spin_iter6-Qwen-14B-Base'
model = QWenLMHeadModelDPO.from_pretrained(model_path, device_map='auto', use_flash_attn=False)
model = model.eval()
disable_dropout(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset_file = f"/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/DPO/dpo_14b_to_72b_spin_iter7.xlsx"
# dataset_file = f"/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/DPO/test.xlsx"
dataset = DPODataset(dataset_file, tokenizer=tokenizer, prompt_col='input', chosen_col='chosen', reject_col='reject')
batch_size=1 
dl = DataLoader(dataset[:], batch_size=batch_size, shuffle=False, collate_fn=get_collate_fn(tokenizer=tokenizer))
result = []

with torch.no_grad():
    for idx, batch in tqdm(enumerate(dl), total=len(dl)//batch_size):
        # if idx < 432:
        #     continue
        batch = {k:v.to(model.device) for k,v in batch.items()}
        batch_logps = model(
            **batch,
            only_reference=True
        )
        result.append(batch_logps)
        # print([i.item() for i in batch_logps])
        

result = torch.cat(result)
for i in range(len(result)):
    ith_tf = result[i]
    save_file({'value':ith_tf}, f'reference_score_caches/{i}.safetensors')
