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
model_path = '/fl-ift/med/jianglei/project/llama-recipes-main/src/checkpoints_dia2abstract2record_0410_348x2_base_17key_claer_72B_35layer_epoch6_b8_2e5_gc0_wd0_seed12345/hf_3'
model = QWenLMHeadModelDPO.from_pretrained(model_path, device_map='auto', use_flash_attn=False)
model = model.eval()
disable_dropout(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset_file = f"/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/dialogue2record_orpo/orpo_train.xlsx"
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
