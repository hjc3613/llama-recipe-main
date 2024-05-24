import torch
from safetensors.torch import load_file
import os
from llama_recipes.utils.supervised_dataset import SupervisedDataset
from transformers import AutoTokenizer

def read_reference_score(indexes):
    reference_score_caches = '/fl-ift/nlp/hujunchao/git_root/llama-recipes-main/src/reference_score_caches'
    result = []
    for i in indexes:
        i_th_tensor = load_file(os.path.join(reference_score_caches, f'{i}.safetensors'))
        result.append(i_th_tensor['value'])
    return torch.stack(result)

tokenizer = AutoTokenizer.from_pretrained('/fl-ift/med/common/Qwen-14B-Base', trust_remote_code=True)
ds = SupervisedDataset('/fl-ift/med/common/datasets/med/tmp.txt', tokenizer=tokenizer, num_process=1)
if __name__ == '__main__':
    # a = read_reference_score([0])
    # b = a.split(1, dim=1)
    # b
    a = 1