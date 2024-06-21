import json
import pandas as pd
import os

def merge_to_jsonl():
    root = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2/all_part_diagnose'
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

if __name__ =='__main__':
    merge_to_jsonl()