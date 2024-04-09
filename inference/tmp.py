import json
import re
import pandas as pd

def convert_json_excel_for_diag2key():
    file = r'/fl-ift/nlp/hujunchao/git_root/sft/data_diag2key/result_dpo_result_retrain_dpo.jsonl'
    # file = r'/fl-ift/nlp/hujunchao/git_root/tmp/data_diag2key/dpo_result.jsonl'
    with open(file, encoding='utf8') as f:
        lines = [json.loads(i) for i in f.readlines()]
    result = []
    for one_person in lines:
        prompt = one_person['dia2key_prefix_prompt']
        content_lst = []
        _id = one_person['id']
        for one_diag in one_person['pred_output']:
            diag_content = one_diag['value']
            pred = one_diag['pred_output']
            gold = one_diag['gold_output']
            result.append({'id':_id, 'diag':diag_content, 'pred':pred, 'gold':gold})
    result = pd.DataFrame.from_dict(result)
    result.to_excel(file.replace('.jsonl', '.xlsx'))


if __name__ == '__main__':
    convert_json_excel_for_diag2key()