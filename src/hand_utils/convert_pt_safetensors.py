import sys
sys.path.append('/fl-ift/nlp/hujunchao/git_root/llama-recipes-main/src')
from llama_recipes.qwen.modeling_qwen import QWenLMHeadModel
from llama_recipes.qwen.configuration_qwen import QWenConfig
import torch
import safetensors
import transformers
pt = '/fl-ift/nlp/hujunchao/git_root/tmp/direct-preference-optimization/local_dirs/hujunchao/dpo_qwen_14b_base_local_data_v3_2024-03-15_05-29-27_685548/step-4992/policy.pt'
sf = '/fl-ift/nlp/hujunchao/models/diag_to_key_dpo_lihui2'
model = QWenLMHeadModel(config=QWenConfig.from_pretrained('/fl-ift/nlp/hujunchao/models/diag_to_key'))
model.load_state_dict(state_dict = torch.load(pt)['state'])

model.save_pretrained(sf)
