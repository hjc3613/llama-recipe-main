# coding=utf-8
from transformers import LlamaTokenizer, AutoTokenizer, LlamaTokenizerFast
from utf8_utils import get_complete_str


file_path = "med_resource.txt"
out_file_path = "resource_out.txt"
model_path = "/data2/skunk/workspace-qwen-14B-chat/triton_models/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
with open(out_file_path, 'w+', encoding='utf-8') as out_file:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            save_format_ids = ""
            line = ''.join(line).strip('\n')
            line = ''.join(line).strip('\r\n')
            #line = "现病史.时间1.主要症状.症状术语:心慌\n现病史.时间1.伴随症状1.症状术语:心悸\n现病史.时间1.阴性症状:无心悸\n"
            #out = 46451|99252|99497|.|20450|16|.|118476|101897|.|108097|29991|.
            prompt_token_ids = tokenizer(line).input_ids
            for id in prompt_token_ids:
                if id == 13:
                    id = '.'
                save_format_ids = save_format_ids + '|' + str(id)
            save_format_ids = save_format_ids[1:] + '|.'
            out_file.write(save_format_ids + '\n')

