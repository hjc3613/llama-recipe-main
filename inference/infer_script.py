from tqdm import tqdm
from os import path, listdir
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,Qwen2ForCausalLM
from transformers import BitsAndBytesConfig
from transformers.generation import GenerationConfig
import os
import sys
import torch
import json
import time
import requests
import pandas as pd
import re
from collections import defaultdict
from argparse import ArgumentParser
from peft import AutoPeftModelForCausalLM
import traceback
import copy
# from merge_all_summary import ReOrderSummary, Method

time_prefix = time.strftime("%Y%m%d",time.localtime(time.time())) 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/fl-ift/med/hujunchao/git_root/llama-recipes-main/src/')
# from llama_recipes.qwen.modeling_qwen import QWenLMHeadModel
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

class DecodeInterface:
    def __init__(self, hf_model_path, tokenizer_name=None) -> None:
        if 'qwen' in hf_model_path.lower() and 'qwen1.5' not in hf_model_path.lower() and 'qwen2' not in hf_model_path.lower():
            flash_attn_args = {
                'use_flash_attn':False
            }
        else:
            flash_attn_args = {
                'attn_implementation':'flash_attention_2'
            }
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=False,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            # use_flash_attn=False,
            **flash_attn_args,
        )
        
        self.generation_config = GenerationConfig.from_pretrained(hf_model_path,trust_remote_code=True)

        if tokenizer_name=='llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
        elif tokenizer_name=='unigpt':
            self.tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=True)

        self.cache = defaultdict(list)

    @torch.no_grad()
    def generate(self, text):
        print('注意当前chat_format=','llama', '请检查是否与训练格式保持一致')
        encoded_input = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
        encoded_input = encoded_input.to(self.model.device)
        generated_ids = self.model.generate(
        **encoded_input,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=self.tokenizer.eos_token_id,
        num_return_sequences=1
        )
        decoded_output = self.tokenizer.decode(generated_ids[0][len(encoded_input['input_ids'][0]):]).replace('</s>', '').replace('<s>', '')
        decoded_output = decoded_output.replace('<|endoftext|>', '').replace('<|im_end|>', '').replace('<|end_of_text|>', '')
        return decoded_output
        

def only_pred_inputoutput(args):
    model_path = os.path.join(args.model_root, args.model_path)
    interface = DecodeInterface(
        hf_model_path=model_path
    )
    if args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    elif args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    result = []
    df = df.iloc[:int(args.record_nums)].fillna('')
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        try:
            inputs:str = row['input']
            # inputs = inputs.replace('否则应补充到模板中：<|im_end|>', '否则应补充到模板中，合并时要分析部位的空间位置关系，同一部位不能出现既正常又异常现象：<|im_end|>')
            
            # messages = [
            #     {'role':'user','content':inputs}
            # ]
            # inputs = interface.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     chat_template=TEMPLATE
            # )
            # inputs  = inputs + '\n<|im_start|>system\n'
            
            
            res = interface.generate(inputs)
        except KeyboardInterrupt:
            traceback.print_exc()
            # print('程序中断')
            sys.exit()
        except:
            traceback.print_exc()
            res = 'error'

        row['pred'] = res
        
        print('='*100)
        print('input\n', inputs)
        print('*'*50)
        print('pred\n', row['pred'])
        print('*'*50)
        print('label\n', row.get('output', ''))
        
        result.append(row)
        # with open(f'{args.file.rsplit(".", maxsplit=1)[0]}_predict.jsonl', mode='a', encoding='utf8') as f:
        #     f.write(json.dumps(row, ensure_ascii=False)+'\n')
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.suffix}.xlsx'
    pd.DataFrame.from_dict(result).to_excel(output_file_excel)



if __name__ == '__main__':
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927梁茜')
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927翟佳逸')
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927邵波')
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--model_root', required=False, default='/fl-ift/med/hujunchao/models/', type=str)
    parser.add_argument('--file', required=True, type=str)
    parser.add_argument('--tokenizer_name', required=False, default=None)
    parser.add_argument('--decode_type', required=False, default='common')
    parser.add_argument('--record_nums', required=False, default=10000)
    parser.add_argument('--stream', required=False, default=False)
    parser.add_argument('--cls_contain', required=False, default=False)
    parser.add_argument('--excel_dir', required=False, default=None)
    parser.add_argument('--input_output', required=False, default=None)
    parser.add_argument('--suffix', required=False, default='predict')
    parser.add_argument('--sheet', required=False, default='Sheet1')
    args = parser.parse_args()
    if args.excel_dir:
        pass
    elif args.input_output:
        only_pred_inputoutput(args)
        # yingxiangbaogao_fenlei(args)
    else:
        pass
