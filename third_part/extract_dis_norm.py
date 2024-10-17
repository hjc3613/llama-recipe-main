import requests
import json
from tqdm import tqdm
import pandas as pd
import re
from collections import Counter

def local_vllm():
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://10.233.125.129:8089/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        temperature=0,
        max_tokens=1024,
        model="qwen110_quant",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""},
            ]
    )
    print("Chat response:", chat_response)

def local_api(text):
    url = r'http://10.233.95.186:7711/chat/completions'
    data = {
            'model':'UniGPT-3.0-MED-RadiologyReport-01',
            'temperature':0,
            # 'prompt':text,
            "messages": [
                {"role":"user", "content":text}
            ],
            'max_tokens':512,
            'repetition_penalty':1,
            'stream':False
        }
    headers = {'Content-Type':'application/json'}
    res = requests.post(url=url, json=data, headers=headers).text
    try:
        res = json.loads(res)
    except:
        print(res)
    res = res['choices'][0]['message']['content']
    return res, res

def extract_only_dis():
    file = r"third_part/dis2abnormal/findings2diags.json"
    with open(file, encoding='utf8') as f:
        data = json.load(f)
    result = []
    
    for suojian, jielun in tqdm(data.items(), total=len(data)):
        p = f'给定放射报告的诊断结论：\n{jielun}\n，请总结出其中的具体疾病或异常表现名字，把可能、疑似、？等不确定性表达去除，只保留具体的疾病或异常表现，以<start>开头，以<end>结束，输出格式为<start>disease or abnormal<end>:'
        _, res = local_api(text=p)
        print(jielun, '<=>', res)
        result.append({'所见':suojian, '结论':jielun, '疾病':res})
        # if len(result) > 100:
        #     break
    result = pd.DataFrame(result)
    result.to_excel(r'third_part/dis2abnormal/诊断生成所见-疾病标准化.xlsx')

def extract_dis():
    file = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/third_part/dis2abnormal/诊断生成所见-疾病标准化.xlsx'
    df = pd.read_excel(file)
    df['疾病_清洗'] = ''
    total_dis = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dis = row['疾病']
        dis_lst = re.findall(r'<start>(.+?)<end>', dis)
        dis = ' | '.join(sorted(dis_lst))
        df.at[idx,'疾病_清洗'] = dis

        total_dis.extend(dis_lst)
    df.to_excel(r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/third_part/dis2abnormal/诊断生成所见-疾病标准化-清洗.xlsx')
    counter = Counter(total_dis).most_common()
    counter = [(item,count) for item, count in counter if count >= 10]
    counter_df = pd.DataFrame.from_records(counter, columns=['dis', 'freq'])
    counter_df.to_excel('/fl-ift/med/hujunchao/git_root/llama-recipes-main/third_part/dis2abnormal/疾病-频次.xlsx')


if __name__ == '__main__':
    # local_api('哈喽')
    # extract_only_dis()
    extract_dis()