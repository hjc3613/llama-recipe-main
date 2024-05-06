import json
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import os

class DataChunk:
    def __init__(self, chunk_size=5000, tokenizer_path='/fl-ift/med/common/Qwen-14B-Base', save_path='chunked.txt') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.chunk_size=chunk_size
        self.save_path=save_path
        self.cache = ''
        self.last_len = 0
    
    def write(self, line):
        f_handler = getattr(self, 'f_handler', None)
        if not f_handler:
            f_handler = open(self.save_path, mode='a', encoding='utf8')
            self.f_handler = f_handler
        f_handler.write(json.dumps({'input':'', 'output':line}, ensure_ascii=False)+'\n')

    def text_token_len(self,text):
        # return len(self.tokenizer.tokenize(text))
        return len(text)

    
    def chunk_and_save(self, text, file):
        if 'drug' in file:
            text_lst = [text] # 药品说明书数据切分
        elif 'med_instruction' in file:
            text_lst = re.split('(\n\n\n\n)', text) # 医疗指令数据切分
        else:
            text_lst = re.split('(.+?[;；。\n!！])', text) # 其它数据切分
        for sub_text in text_lst:
            if not sub_text:
                continue
            if self.text_token_len(sub_text) + self.last_len > self.chunk_size and len(self.cache)>500:
                self.write(self.cache)
                self.cache = sub_text
                self.last_len = self.text_token_len(sub_text)
            else:
                self.cache+=sub_text
                self.last_len+=self.text_token_len(sub_text)


def main(root):
    # 通用
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/books/150KBooks.mobi.jsonl'
    # data_chunk = DataChunk(save_path='preprocessed2/book.txt')
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/wiki/wiki.qc.fix2'
    # data_chunk = DataChunk(save_path='preprocessed2/wiki.txt')
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/C4/c4.qc'
    # data_chunk = DataChunk(save_path='preprocessed2/c4.txt')
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/toutiao/toutiao.qc'
    # data_chunk = DataChunk(save_path='preprocessed2/toutiao.txt')
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/zhihu/zhihu.qc.fix'
    # data_chunk = DataChunk(save_path='preprocessed2/zhihu.txt')
    # file = r'/baykal/unisound/lihui/pretrain_data/data_1111/CommonCrawl/cc.qc'
    # data_chunk = DataChunk(save_path='preprocessed2/commoncrawl.txt')
    # file = r'/baykal/unisound/lihui/bak/data/genearl_qa/Tulu_V2_Mix/merge.jsonl'
    # data_chunk = DataChunk(save_path='preprocessed2/tulu.txt', chunk_size=8000)

    # 医疗
    
    # file = '/baykal/unisound/lihui/bak/unsup_data_med/med-ptr-v0.4/baidubaike.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/baidubaike_med.txt')
    # file = '/baykal/unisound/lihui/bak/unsup_data_med/v0.12_wanfang/out.jsonl'
    # data_chunk = DataChunk(save_path='preprocessed2/wanfang12.txt')
    # file = '/baykal/unisound/lihui/bak/unsup_data_med/v0.13_wanfang/out.jsonl'
    # data_chunk = DataChunk(save_path='preprocessed2/wanfang13.txt')
    # file = '/baykal/public/medical_spider/丁香园/20240424.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/dingxiangyuan.txt')
    # file = '/baykal/public/medical_spider/中国医药查询平台/20240423.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/zhongyiyao0423.txt')
    # file = '/baykal/public/medical_spider/中国医药查询平台/20240424.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/zhongyiyao0424.txt')
    # file = '/baykal/public/medical_spider/中国医药查询平台/20240425.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/zhongyiyao0425.txt')
    # file = '/baykal/public/medical_spider/默沙东诊疗手册/20240423.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/moshadong0423.txt')
    # file = '/baykal/public/medical_spider/默沙东诊疗手册/20240424.txt.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/moshadong0424.txt')
    # file=r'/data/hujunchao/git_root/LLM_dataset/train_drug_text.jsonl'
    # data_chunk = DataChunk(save_path='preprocessed2/drugs.txt')
    # file = 'instructions_origin.txt'
    # data_chunk = DataChunk(save_path='preprocessed2/instructions.txt')
    files = [i for i in os.listdir(root) if i.endswith('.jsonl')]
    print('files to preprocessed: ', '\n'.join(files))
    for file in tqdm(files):
        full_path = os.path.join(root, file)
        data_chunk = DataChunk(save_path=f'{root}/chunked/{file.split(".")[0]}.jsonl', chunk_size=(6000 if 'english_instruction' in file else 3000))
        with open(file=full_path, encoding='utf8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                if 'book' in file:
                    text = data['content']
                elif 'drug' in file:
                    text = data['output']
                else:
                    text = '\n'.join(data['paras'])
                data_chunk.chunk_and_save(text, file=file)

if __name__ == '__main__':
    main('/fl-ift/med/common/datasets/med/mix_med_common/version01')