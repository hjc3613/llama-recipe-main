import json
import pandas as pd
import os
import re

def merge_to_jsonl():
    root = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2/all_part_diagnose_fix_xuhao'
    excels = os.listdir(root)
    result = []
    for excel in excels:
        path = os.path.join(root, excel)
        df = pd.read_excel(path)
        df = df.sample(n=50, random_state=100)
        rows = [dict(row) for idx, row in df.iterrows()]
        texts = [{'text':row['input']+row['output']} for row in rows]
        result.extend(texts)
    result = [json.dumps(i, ensure_ascii=False) for i in result]
    with open('/fl-ift/med/hujunchao/git_root/quant_awq/quant2.jsonl', mode='w', encoding='utf8') as f:
        f.write('\n'.join(result))

def merge_to_jsonl_recursive():
    root = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology'
    subroots = [
        ('4_part_ct', 20),
        ('all_part_diagnose_fix_xuhao_aligned_train_instruct', 50),
        ('end2end_template_report', 35)
    ]
    result = []
    for subroot, sample_nums in subroots:
        subpath = os.path.join(root, subroot)
        files = os.listdir(subpath)
        for file in files:
            if '副本' in file:
                continue
            if file == 'end2end_multi_template_train.xlsx':
                sample_nums += 100
            path = os.path.join(subpath, file)
            df = pd.read_excel(path)
            df = df.sample(sample_nums, random_state=100)
            rows = [dict(row) for idx, row in df.iterrows()]
            texts = [{'text':row['input']+row['output']} for row in rows]
            result.extend(texts)
    result = [json.dumps(i, ensure_ascii=False) for i in result]
    with open('/fl-ift/med/hujunchao/git_root/quant_awq/quant3.jsonl', mode='w', encoding='utf8') as f:
        f.write('\n'.join(result))
if __name__ =='__main__':
    # merge_to_jsonl()
    merge_to_jsonl_recursive()