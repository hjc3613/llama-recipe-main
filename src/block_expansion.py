import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from copy import deepcopy
import pdb

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='/fl-ift/med/common/Qwen-7B-Chat', type=str, help="original model path")
    parser.add_argument("--output_path", default='pytorch_model.bin', type=str, help="deepened model ckpt save path")
    parser.add_argument("--original_layers", default=32, type=int, help="original model num layers")
    parser.add_argument("--layers", default=40, type=int, help="deepen model num layers")

    # Parse the arguments
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    ckpt = model.state_dict()
    
    pdb.set_trace()

    split = int(args.original_layers / (args.layers - args.original_layers))
    layer_cnt = 0

    output = {}
    for i in range(args.original_layers):
        for k in ckpt:
            if ('h.' + str(i) + '.') in k:
                output[k.replace(('h.' + str(i) + '.'), ('h.' + str(layer_cnt) + '.'))] = ckpt[k]
                pdb.set_trace()
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('h.' + str(i) + '.') in k:
                    if 'down_proj' in k or 'c_proj' in k:
                        output[k.replace(('h.' + str(i) + '.'), ('h.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                    else:
                        output[k.replace(('h.' + str(i) + '.'), ('h.' + str(layer_cnt) + '.'))] = ckpt[k]


            layer_cnt += 1
        
    assert layer_cnt==args.layers
    ###'lm_head.weight', 'transformer.ln_f.weight', 'transformer.wte.weight'
    for k in ckpt:
        if not 'h.' in k:
            output[k] = ckpt[k]

    pdb.set_trace()
    torch.save(output, args.output_path)

@torch.no_grad()
def main2():
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='/fl-ift/med/common/Qwen-14B-Base', type=str, help="original model path")
    parser.add_argument("--output_path", default='pytorch_model.bin', type=str, help="deepened model ckpt save path")
    parser.add_argument("--group_size", default=5, type=int, help="group size when split layers")

    # Parse the arguments
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    qw = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    qw2 = AutoModelForCausalLM.from_pretrained('/fl-ift/med/hujunchao/models/qw_pro', torch_dtype=torch.float16, trust_remote_code=True)
    h = qw.transformer.h
    new_h = nn.ModuleList()
    for idx, model in enumerate(h, start=1):
        if idx % args.group_size == 0:
            model_expand = deepcopy(model)
            model_expand.attn.c_proj.weight.zero_()
            model_expand.mlp.c_proj.weight.zero_()
            
            new_h.append(model)
            new_h.append(model_expand)
        else:
            new_h.append(model)
    qw.transformer.h = new_h
    qw.save_pretrained(args.model_path+'_pro')
    tokenizer.save_pretrained(args.model_path+'_pro')

if __name__ == "__main__":
    main2()
