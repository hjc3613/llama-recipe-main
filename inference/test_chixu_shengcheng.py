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
from llama_recipes.qwen.modeling_qwen import QWenLMHeadModel

# hf_model_path = '/data/hujunchao/models/learn_gpt4_continue_gen_no_blank/checkpoint-25'
# tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
# model = UniGPTForCausalLM.from_pretrained(hf_model_path)
# model = UniGPTForCausalLM.from_pretrained(
# hf_model_path,
# torch_dtype=torch.bfloat16,
# low_cpu_mem_usage=True,
# device_map="auto",
# load_in_8bit=False
# )

def to_jsonl(dct_lst, path):
    with open(path, encoding='utf8', mode='w') as f:
        dct_lst = [json.dumps(i, ensure_ascii=False) for i in dct_lst]
        f.write('\n'.join(dct_lst)+'\n')

class PostProcess:
    def __init__(self) -> None:
        self.han2digit_map = {
            '幺':1,
            '一':1,
            '二':2,
            '三':3,
            '四':4,
            '五':5,
            '六':6,
            '七':7,
            '八':8,
            '九':9,
            '零':0,
        }

    def tizhong(self, s):
        result = (None, -1)
        pats = [
            re.compile(r'((?:[一二]百)?[一二三四五六七八九]十[一二三四五六七八九多])(斤|公斤|千克)'), # 一百五十八斤、五十八公斤
            re.compile(r'((?:[一二]百)?[一二三四五六七八九]{2}十)(斤)'), # 一百三四十斤
            re.compile(r'((?:[一二]百)?[一二三四五六七八九]十)(斤|公斤|千克)'), # 一百三十斤
            re.compile(r'([四五六七八九]{2}十)(公斤|千克)'), # 五六十公斤
            re.compile(r'((?:[一二]百)零[一二三四五六七八九])(斤|公斤|千克)') # 一百零五斤
        ]
        for pat in pats:
            finded_lst = list(re.finditer(pat, s))
            for finded in finded_lst:
                span = finded.span()
                if span[1] > result[1]:
                    result = ((finded.group(1), finded.group(2)), span[1])
        result = result[0]
        if result:
            han, unit = result
            digital = self.han2digital(han)
            if unit == '斤':
                digital /= 2
            result = f'{digital}kg'
        return result
    
    def shengao(self, s):
        result = (None, -1, -1)
        pats = [
            re.compile(r'身高.{0,5}([一幺])([四五六七八九])([一二三四五六七八九零])'),
            re.compile(r'(一)米([四五六七八九])([一二三四五六七八九零])'),
            re.compile(r'(一)米([四五六七八九])'),
        ]
        for idx, pat in enumerate(pats):
            finded_lst = list(re.finditer(pat, s))
            for finded in finded_lst:
                span = finded.span()
                if span[1] > result[1]:
                    result = (finded, span[1], idx)
        result, pat_idx = result[0], result[2]
        if result:
            if pat_idx == 0 or pat_idx == 1:
                bai = self.han2digit_map[result.group(1)]
                shi = self.han2digit_map[result.group(2)]
                ge = self.han2digit_map[result.group(3)]
                
            elif pat_idx == 2:
                bai = self.han2digit_map[result.group(1)]
                shi = self.han2digit_map[result.group(2)]
                ge = 0
            else:
                bai = '-'
                shi = '-'
                ge = '-'
            result = f'{bai}{shi}{ge}cm'
        return result

    def han2digital(self, han):
        bai = re.findall('[一二三四五六七八九](?=百)', han)
        shi = re.findall('[一二三四五六七八九](?=十)', han)
        ge = re.findall(r'(?<=[零十])[一二三四五六七八九]', han)
        result = 0
        if bai:
            result += self.han2digit_map[bai[0]]*100
        if shi:
            result += self.han2digit_map[shi[0]]*10
        if ge:
            result += self.han2digit_map[ge[0]]
        return result

    def process(self, output, dialogue):
        if not re.search('身高|体重', output):
            return output
        output_lst= [re.split(r':|：', i, maxsplit=1) for i in output.split('\n') if re.search(r':|：', i.strip())]
        output_dict = dict(zip(*zip(*output_lst)))
        result = {}
        for k,v in output_dict.items():
            if re.search(r'体格检查\.体重', k):
                tizhong = self.tizhong(dialogue)
                if tizhong:
                    result[k] = tizhong
                else:
                    result[k] = v
            elif re.search(r'体格检查\.身高', k):
                shengao = self.shengao(dialogue)
                if shengao:
                    result[k] = shengao
                else:
                    result[k] = v
            else:
                result[k] = v
        result = [f'{k}:{v}' for k,v in result.items()]
        return '\n'.join(result)
# tmp = PostProcess()
# file = r'C:\Users\YZS\Desktop\持续生成测试集\baichuan_histrounds15_filteroutthreash0.06_fuhe_0925_0926\train_tige_common_streamNone_0.06_tizhong.xlsx'
# items = list(pd.read_excel(file)['当前对话'])
# items_result = []
# for item in items:
#     if '一百二三十斤吧' in item:
#         a = 1
#     tizhong = tmp.tizhong(item)
#     shengao = tmp.shengao(item)
#     items_result.append((item, tizhong, shengao))
# pd.DataFrame.from_records(items_result).to_excel(file.replace('.xlsx', '_tmp.xlsx'))
# a=1

def dialogue_contains_summary(dialogue, abstract):
    return '是'
    url = "http://10.10.20.40:7013/chat/completions" # abs post process
    prompt = f'判断下面的信息摘要和医患对话是否匹配或者蕴含，输出是或者否：\n医患对话：\n{dialogue}\n信息摘要：\n{abstract}\n'
    payload = json.dumps({
        "model": "UniGPT2.07a.AbstractClass.00.01",
    "temperature": 0,
    "prompt": prompt,
    "max_tokens": 5,
    "repetition_penalty": 1,
    "stream": False
    })
    res = requests.post(url=url, data=payload).json()['choices'][0]['message']['content']
    return res

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps = per_token_logps[0].tolist()
    tokenids = labels[0].tolist()
    return logps, tokenids
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
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

        # self.reorder_summary = ReOrderSummary(
        #     merge_regular=os.path.join(PROJECT_ROOT, 'all_summary_keys', 'merge_regular.tsv'),
        #     key_positions=os.path.join(PROJECT_ROOT, 'all_summary_keys', 'key_position.txt'),
        #     gensim_model_path=r'E:\bert_models\chinese_word_vector\sgns.baidubaike.bigram-char.bz2', # Method.gensim 时有效
        #     similary_method=Method.fuzzywuzzy, # 
        #     regex_file = os.path.join(PROJECT_ROOT, 'all_summary_keys', 'regular_match.txt')
        # )
        self.post_process = PostProcess()

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

    @torch.no_grad()
    def generate_old(self, text):
        # text = f'{text}Response:'
        generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            # early_stopping=True,
            max_new_tokens=256,
            chat_format='raw',
            eos_token_id = 151643,
            pad_token_id=151643,
            max_window_size=6144,
        )
        print('注意当前chat_format=',generation_config.chat_format, '请检查是否与训练格式保持一致')
        response = self.model.chat(self.tokenizer, text, history=[], append_history=False, generation_config=generation_config)
        return response[0]

    @torch.no_grad()
    def forward(self, text, example_id):
        import matplotlib.pylab as plt
        # def bi_score(x, y):
        #     a = hidden_states[x].squeeze(0).to('cuda:0') # [seq_len, hidden size]
        #     b = hidden_states[y].squeeze(0).to('cuda:0') # [seq_len, hidden size]
        #     b = b.to(a.device)
        #     bi_each_dim = (a*b).sum(0)/(torch.norm(a, p=2, dim=0) * torch.norm(b, p=2, dim=0)) # [hidden size]
        #     bi_one_layer = bi_each_dim.mean()
        #     bi_final = 1 - bi_one_layer
        #     return bi_final
        batch = self.tokenizer(text=text, return_tensors='pt')
        batch = {k:v.to(self.model.device) for k,v in batch.items()}
        res = self.model(**batch, )
        # hidden_states = res.hidden_states
        # for layer_idx in range(len(hidden_states)-2):
        #     bi = bi_score(layer_idx, layer_idx+2)
        #     if layer_idx in [33,34,35,36]:
        #         tmp = 1
        #     with open('./bi_score_tmp3.txt', mode='a', encoding='utf8') as f:
        #         f.write(f'{example_id},{layer_idx},{bi.item()}'+'\n')

        # for layer, attn in enumerate(res.attentions[:5]):
        #     plt.imshow(1- attn[0][0].cpu().to(torch.float32).numpy(), cmap='hot', interpolation=None)
        #     plt.savefig(f'tmp/{example_id}-{layer}.png')
        res
        logps, tokenids = _get_batch_logps(res.logits, batch['input_ids'])
        tokens = self.tokenizer.convert_ids_to_tokens(tokenids)
        token_probs = list(zip([i.decode() for i in tokens], logps))
        return token_probs
    @torch.no_grad()
    def generate_qw(self, text, admission_date):
        generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            # early_stopping=True,
            max_new_tokens=512,
            chat_format='raw',
            eos_token_id = 151643,
            pad_token_id=151643,
            max_window_size=6144,
        )
        print('注意当前chat_format=',generation_config.chat_format, '请检查是否与训练格式保持一致')
        response = self.model.chat(self.tokenizer, text, history=[], append_history=False, generation_config=generation_config)
        # response_fixed = self.reorder_summary.uniform_kv_summary(response[0], admission_date)
        response_fixed = response[0]
        return response[0], response_fixed

    def format_cache_to_input(self, record_id, window=10, stream=False):
        caches_of_record_id = self.cache.get(record_id, [])
        if stream:
            print('stream 方式组合历史摘要')
            # 当缓存中的对话轮数大于window时，需将新的window窗口外的对话摘要放进历史摘要中
            if len(caches_of_record_id) > window:
                _, _, new_poped_summary, _ = caches_of_record_id[-(window+1)]
                if re.search('当前对话中', new_poped_summary):
                    new_poped_summary = ''
            else:
                new_poped_summary = ''
            # 截至当前为止所有的历史摘要
            pre_summary = caches_of_record_id[-1][-1] if len(caches_of_record_id)>0 else ''
            pre_summary = f'{pre_summary}\n{new_poped_summary}'.strip()
        else:
            print('非stream方式组合历史摘要')
            pre_summary = '\n'.join(i[2] for i in caches_of_record_id[:-window] if not re.search(r'当前对话中',i[2]))
        pre_summary = re.sub(r'\n+', '\n', pre_summary)
        pre_summary = self.reorder_summary.post_process_abs(pre_summary)
        pre_summary_ = '历史所有结论:\n' + pre_summary
        result = []
        result.append(pre_summary_)
        for round, pre_diag, cur_summary, _ in caches_of_record_id[-window:]:
            result.append(f'{pre_diag}\n结论:{cur_summary or "当前对话中无法得到确定性信息"}')
        return '\n'.join(result), pre_summary
    
    def format_cache_to_input_jianglei(self, record_id, window=10, stream=False):
        caches_of_record_id = self.cache.get(record_id, [])
        pre_summary = '\n'.join(i[2] for i in caches_of_record_id[:-window] if not re.search(r'当前对话中',i[2]))
        pre_summary = re.sub(r'\n+', '\n', pre_summary)
        # pre_summary = self.reorder_summary.post_process_abs(pre_summary)
        pre_summary_ = '历史信息摘要：\n' + pre_summary
        result = []
        result.append(pre_summary_)
        result.append('历史医患对话及摘要：')
        for round, pre_diag, cur_summary, _ in caches_of_record_id[-window:]:
            result.append(f'医患对话：\n{pre_diag}\n信息摘要：\n{cur_summary or "当前对话无有效医学摘要信息"}\n')
        return '\n'.join(result), pre_summary

    def iter_generate(self, cur, record_id, round, date,stream):
        inputs, pre_summary = self.format_cache_to_input_jianglei(record_id, window=10, stream=stream)
        # inputs = f'就诊日期:{date}\n{inputs}\n{cur}\n结论:'
        inputs = f'根据下面的历史摘要信息和历史医患对话及摘要，生成最新轮次对话的医学信息摘要：\n{inputs}\n医患对话：\n{cur}\n信息摘要：\n'
        res_fixed = self.generate(inputs)
        # res_org, res_fixed = res_org.strip(), res_fixed.strip()
        self.cache[record_id].append((round, cur, res_fixed, pre_summary))
        return res_fixed, res_fixed, inputs
    
    def get_final_summary(self, record_id):
        caches_of_record_id = self.cache.get(record_id, [])
        final_summary = '\n'.join(i[2] for i in caches_of_record_id if not re.search(r'当前对话中',i[2]))
        final_summary = re.sub(r'\n+', '\n', final_summary)
        final_summary = self.reorder_summary.post_process_abs(final_summary)
        return final_summary
    
    def process(self, row, type,stream=False):
        if type=='common':
            return *self.generate(row['input'], row['admission_date']), row['input']
        elif type=='iter':
            return self.iter_generate(row['当前对话'], row['record_id'], row.get('round', -1), row.get('', 'admission_date'),stream)
        else:
            raise Exception('type 传值错误')
        
def process_dir(root, args):
    excels = [i for i in os.listdir(root) if i.endswith('.xlsx') and not re.search(r'_tmp|_abnormal', i)][:]
    # exists = [i.replace('_预标_qwen', '') for i in os.listdir(root) if i.endswith('.xlsx') and re.search(r'_tmp|预标', i)]
    # excels = [i for i in excels if i not in exists]
    interface = DecodeInterface(
        hf_model_path=args.model_path
    )
    for excel in tqdm(excels):
        path = os.path.join(root, excel)
        df = pd.read_excel(path, sheet_name='Sheet5')
        df = df.sample(n=200, random_state=100).reset_index(drop=True)
        result = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            row = dict(row)
            inputs = row['input']
            prompt = f'请找出下列影像学报告中检查所见的异常所见，忽略阴性描述\n检查所见：{inputs}\n异常所见：'
            res = interface.generate(prompt)
            row['pred'] = res
            print('='*100)
            print('input\n', prompt)
            print('*'*50)
            print('pred\n', row['pred'])
            
            result.append({**row})
            
        result = pd.DataFrame.from_dict(result)
        result.to_excel(path.replace('.xlsx', '_abnormal.xlsx'))

def only_pred_inputoutput(args):
    interface = DecodeInterface(
        hf_model_path=args.model_path
    )
    if args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    elif args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    result = []
    df = df.iloc[:int(args.record_nums)]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        try:
            inputs = row['input']
            # messages = [
            #     # {"role": "system", "content": "You are a helpful assistant."},
            #     # {'role':'user', 'content':row['dialogue']+'\n基于上述对话，通过问答方式对对话做总结，包括但不限于患者的症状、疾病、用药、手术、血糖指标、血压指标等详信息则要尽量详细复述出来，这些总结内容要满足能生成一份高质量的病历报告：'}
            #     {'role':'user', 'content':row['dialogue']+'\nplease generate an standart medical record correspond to the above dialogue, including chief complaint、history of present illness、history of past illness、family history、Physical examination、treatment opinions, result in english：'}
            # ]
            # inputs = interface.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # inputs = f'{row["dialogue"]}\n请结合上述对话，总结出对话中出现的问题和答案:'
            # interface.forward('''        头颅形态如常，脑实质内无明显异常密度区及形态改变区。脑室系统、脑沟、裂、池无异常。中线结构居中。颅骨未见明确骨折征象。额部头皮软组织密度增高。\n请根据上述影像报告，生成诊断结论: 1、头颅CT平扫脑内未见明显异常；  2(Somehow I don't like the word "increased" in the report, so I will use "edema" instead. )额部头皮软组织肿胀。''', idx)
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

def only_pred_inputoutput2(args):
    interface = DecodeInterface(
        hf_model_path=args.model_path
    )
    if args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    elif args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    result = []
    df = df.iloc[:int(args.record_nums)]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        try:
            # inputs1 = f"{row['dialogue']}\n请结合上述医患对话和患者就诊日期：{row['admisstion_date']}, 生成医患对话的关键信息摘要："
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":f"{row['dialogue']}\n请结合上述医患对话和患者就诊日期：{row['admisstion_date']}, 生成医患对话的关键信息摘要："},
                # {'role':'user', 'content':row['dialogue']+'\n基于上述对话，通过问答方式对对话做总结，包括但不限于患者的症状、疾病、用药、手术、血糖指标、血压指标等详信息则要尽量详细复述出来，这些总结内容要满足能生成一份高质量的病历报告：'}
            ]
            inputs2 = interface.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res = interface.generate(inputs2)
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":f"{row['dialogue']}\n请结合上述医患对话和患者就诊日期：{row['admisstion_date']}, 生成医患对话的关键信息摘要："},
                # {'role':'user', 'content':row['dialogue']+'\n基于上述对话，通过问答方式对对话做总结，包括但不限于患者的症状、疾病、用药、手术、血糖指标、血压指标等详信息则要尽量详细复述出来，这些总结内容要满足能生成一份高质量的病历报告：'},
                {'role':'assistant', 'content':res},
                # {'role':'user', 'content':'请基于上述对话和总结的核心问题与答案，写出一份高质量电子病历：'},
                {'role':'user', 'content':'请结合上述医患对话、关键摘要信息和患者就诊日期生成标准的电子病历：'},
            ]
            inputs3 = interface.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # inputs = f'{inputs}\n{res}\n请结合上述医患对话、关键摘要信息和患者就诊日期生成标准的电子病历：'
            res = interface.generate(inputs3)
            # inputs = f'{row["dialogue"]}\n请你总结上述对话，分析医生的提问和患者的回答，生成一份一份病历报告：'
            # res = interface.generate_old(inputs)
            # inputs = f"{row['dialogue']}\n请结合上述对话，总结出对话中出现的问题和答案:"
            # res = interface.generate_old(inputs)
            # row['pred_nlp_summary'] = res
            # inputs = f'{inputs}\n上述医患对话中的关键问题和答案如下：\n{res}\n请基于对话，结合关键问题和答案，生成一份病历：'
            # res = interface.generate_old(inputs)
        except KeyboardInterrupt:
            traceback.print_exc()
            # print('程序中断')
            sys.exit()
        except:
            traceback.print_exc()
            print(inputs3)
            res = 'error'

        row['pred'] = res
        row['input_new'] = inputs3
        print('='*100)
        print('input\n', inputs3)
        print('*'*50)
        print('pred\n', row['pred'])
        print('*'*50)
        print('label\n', row.get('output', ''))
        
        result.append(row)
        # with open(f'{args.file.rsplit(".", maxsplit=1)[0]}_predict.jsonl', mode='a', encoding='utf8') as f:
        #     f.write(json.dumps(row, ensure_ascii=False)+'\n')
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.suffix}.xlsx'
    pd.DataFrame.from_dict(result).to_excel(output_file_excel)

def yingxiangbaogao_fenlei(args):
    TEMPLATE = {
    '0':'无意义的前缀内容',
    '1':'胸廓两侧对称',
    '2':'支气管血管束清晰',
    '3':'肺内未见明确异常密度灶',
    '4':'气管、双肺支气管及其分支管腔通畅',
    '5':'双侧肺门及纵隔内未见明显增大淋巴结',
    '6':'心脏形态、大小正常',
    '7':'胸膜未见明确病变',
    '8':'胸廓骨质及软组织未见明显异常',
    '9':'其他异常'
    }   
    TEMPLATE2 = {v:k for k,v in TEMPLATE.items()}
    interface = DecodeInterface(
        hf_model_path=args.model_path
    )
    if args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    elif args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    result = []
    df = df.iloc[:int(args.record_nums)]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        try:
            abnormals:str = row['pred']
            result_ = []
            tmp_result_ = []
            for item in abnormals.split('\n'):
                item = item.split('、', maxsplit=1)[1]
                template = '\n'.join([f'.{v}' for k,v in TEMPLATE.items()])
                inputs = f'给定模板：\n{template}\n请找出“{item}”与上述模板中最匹配的一项并判断二者是否矛盾：'
                res = interface.generate(inputs)
                tmp_result_.append(res)
                parsed = re.findall(r'最匹配项：(.+?),二者(不矛盾|矛盾)', res)
                q_cls, maodun = parsed[0]
                num = TEMPLATE2[q_cls]
                if '不' in maodun:
                    num = '+'+num
                result_.append(f'{num}={item}')
            res = '\n'.join(result_)
            tmp_res = '\n'.join(tmp_result_)
        except KeyboardInterrupt:
            traceback.print_exc()
            # print('程序中断')
            sys.exit()
        except:
            traceback.print_exc()
            res = 'error'

        row['pred2'] = res
        row['tmp'] = tmp_res
        print('='*100)
        print('input\n', inputs)
        print('*'*50)
        print('pred\n', row['pred2'])
        print('*'*50)
        print('label\n', row.get('output', ''))
        
        result.append(row)
        # with open(f'{args.file.rsplit(".", maxsplit=1)[0]}_predict.jsonl', mode='a', encoding='utf8') as f:
        #     f.write(json.dumps(row, ensure_ascii=False)+'\n')
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.suffix}.xlsx'
    pd.DataFrame.from_dict(result).to_excel(output_file_excel)

def _stream_dialogue_by_role(dialogue, round=2):
    if isinstance(dialogue, str):
        dialogue_lst = dialogue.split('\n')
    elif isinstance(dialogue, list):
        dialogue_lst = dialogue
    dialogue_lst = [i for i in dialogue_lst if i]
    result = [[]]
    last_line = ''
    for line in dialogue_lst:
        if '医生' in line[:5] and len(result[-1])>=2*round and '患者' in last_line[:5]:
            result.append([line])
        else:
            result[-1].append(line)
        last_line = line
    return ['\n'.join(i) for i in result]

def dialogue2nlp_abs(args):
    interface = DecodeInterface(
        hf_model_path=args.model_path
    )
    if args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    elif args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file)
    result = []
    df = df.iloc[:int(args.record_nums)]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        dialogue = row['input']
        dialogue_lst = _stream_dialogue_by_role(dialogue, round=1)
        total_abs = ''
        for i in range(0, len(dialogue_lst), 6):
            input = '\n'.join(dialogue_lst[max(0, i-2):i+6])
            input = f'{input}\n对上述对话做总结：'
            pred = interface.generate_old(input)
            row['pred'] = pred
            print('='*100)
            print('input\n', input)
            print('*'*50)
            print('output\n', pred)
            total_abs += pred+'\n'
            result.append(row)
        row['total_abs'] = total_abs
        result.append(row)
    df = pd.DataFrame.from_dict(result)
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.suffix}.xlsx'
    df.to_excel(output_file_excel)

def duplicate_keys_for_diag2abs(pred_output_list):
    new_output_dict = {}
    new_output = []
    pred_output_list = list(dict.fromkeys(pred_output_list).keys())
    for pred in pred_output_list:
        if pred == "当前对话无有效医学摘要信息": continue
        # pred = pred.replace("，", "、")
        # if pred.count(":") == 1:
        #     key, value = pred.split(":")
        #     if key not in new_output_dict.keys():
        #         new_output_dict.update({key: [value]})
        #     else:
        #         new_output_dict[key].append(value)
        pred = pred.replace(":", "：")
        items = pred.split("：")
        key = items[0]
        value = ":".join(items[1:])
        if key not in new_output_dict.keys():
            new_output_dict.update({key: [value]})
        else:
            new_output_dict[key].append(value)
    # 子串去重
    for key, value_list in new_output_dict.items():
        value_list = list(dict.fromkeys(value_list).keys())
        new_value = copy.deepcopy(value_list)
        for i in range(len(value_list)):
            for j in range(len(value_list)):
                if i == j: continue
                if value_list[i].find(value_list[j]) != -1:
                    if value_list[j] in new_value:
                        new_value.remove(value_list[j])
        # print(new_value)
        new_output.append(key + ":" + "，".join(new_value))
    # 根据资源key进行重新排序
    f = open("./all_summary_keys/key_position.txt", 'r', encoding='utf-8')
    key_order = f.readlines()
    key_order = [key.strip("\n") for key in key_order]
    f.close()
    new_abs_list = []
    for abs in new_output:
        abs_list = re.split(":|：", abs)
        new_abs_list.append({"key": abs_list[0].replace(" ", ""), "value": "，".join(abs_list[1:]).replace(" ", "")})
    # key根据固定排序
    sorted_items = sorted(new_abs_list,
                          key=lambda x: key_order.index(x['key']) if x['key'] in key_order else float('inf'))
    new_abs_list = []
    for items in sorted_items:
        new_abs_list.append(items["key"] + "：" + items["value"])
    return '\n'.join(new_abs_list)

def main(args):
    interface = DecodeInterface(args.model_path, args.tokenizer_name)
    if args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file)
    elif args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    else:
        raise Exception('只支持xlsx和jsonl文件')
    df = df.fillna('')
    # if 'admission_date' not in df:
    #     df['admission_date'] = df['input'].str.findall('(?<=就诊日期:)(.+?)\n').str[0]
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.decode_type}_stream{args.stream}.xlsx'
    output_file_jsonl = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.decode_type}_stream{args.stream}.jsonl'
    output_file_final = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.decode_type}_stream{args.stream}_final.xlsx'
    print(f'ready to save to {output_file_excel} and {output_file_jsonl}')
    result = []
    result_final = []
    processed_record_num = set()
    cur_dialogue_cache = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if 'record_id' not in row:
            row['record_id'] = row['record_name'].split('_')[0]
        if idx==125:
            a = 1
        processed_record_num.add(row['record_id'])
        if len(processed_record_num) > int(args.record_nums):
            break
        row = dict(row)
        cur_dialogue_cache[row['record_id']].append(row['当前对话'])
        res_org, res, inputs = interface.process(row, type=args.decode_type, stream=eval(args.stream))
        if '无法得到确定性信息' in res:
            res = ''
        # res = interface.post_process.process(res, row['当前对话'])
        row['pred_output'] = res
        row['pred_output_org'] = res_org
        row['input_new'] = inputs

        print(f'pred output:\n{res}')
        print(f'label: \n{row.get("output") or row.get("label")}')

        last_n_cur_dialogue = '\n'.join(cur_dialogue_cache.get(row['record_id'])[-9:])
        cls_contain = dialogue_contains_summary(dialogue=last_n_cur_dialogue, abstract=res)
        row['蕴含模型输出'] = cls_contain
        result.append({**row})

    result = pd.DataFrame.from_dict(result)
    if 'output' in result.columns:
        result['gold_output'] = result['output']
    else:
        result['gold_output'] = result['label']
    # result.drop('output', axis=1, inplace=True)
    # if 'id' not in result.columns:
    #     result['id'] = result['record_id']+'_'+result['round'].astype(str)
    print(f'save to {output_file_excel}')
    result.to_excel(output_file_excel)
    # print(f'save to {output_file_jsonl}')
    # to_jsonl(result.to_dict(orient='records'), output_file_jsonl)
    for record_id, subdf in result.groupby('record_id'):
        all_summary_pred = [j for i in subdf['pred_output'] if i and not re.search(r'当前对话', i) for j in i.split('\n')]
        all_summary_pred = duplicate_keys_for_diag2abs(all_summary_pred)

        subdf_cls_contain = subdf[subdf['蕴含模型输出'] == '是']
        all_summary_pred_cls_contain = [j for i in subdf_cls_contain['pred_output'] if i and not re.search(r'当前对话', i) for j in i.split('\n')]
        all_summary_pred_cls_contain = duplicate_keys_for_diag2abs(all_summary_pred_cls_contain)
        
        all_summary_label = [j for i in subdf['gold_output'] if i and not re.search(r'当前对话', i) for j in i.split('\n')]
        all_summary_label = duplicate_keys_for_diag2abs(all_summary_label)
        depart = list(subdf['depart'])[0] if 'depart' in subdf.columns else list(subdf['department'])[0]
        dialogue = '\n'.join([i for i in subdf['当前对话']])
        result_final.append(
            {'record_name':record_id, 
             'pred_output':all_summary_pred, 
             'pred_output_cls_contain':all_summary_pred_cls_contain,
             'label':all_summary_label, 
             'dialogue':dialogue, 
             'department':depart,
            #  'pred_output_gpt4':all_summary_gpt4
             })
    print(f'save to {output_file_final}')
    pd.DataFrame.from_dict(result_final).to_excel(output_file_final)
if __name__ == '__main__':
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927梁茜')
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927翟佳逸')
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/0927数据标注质量验证/20230927邵波')
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
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
        process_dir(args.excel_dir, args)
    elif args.input_output:
        only_pred_inputoutput(args)
        # yingxiangbaogao_fenlei(args)
    else:
        main(args)
