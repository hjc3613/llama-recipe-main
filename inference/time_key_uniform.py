import pandas as pd
import os
import sys
import re
import json
from datetime import datetime
from tqdm import tqdm

class TimeValueConvert:
    def __init__(self) -> None:
        self.reg_pipes = self.get_regex_pipeline()
        self.admission_date = None
        pass

    def convert_y(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_y = int(matched)
        result = f'{adm_y-v_y}年'
        return result

    def convert_m(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_m = int(matched)
        if adm_m-v_m>0:
            result = f'{adm_m-v_m}个月'
        else:
            result = None
        return result

    def convert_w_nums(self, matched):
        v_w = matched
        result = f'{v_w}周'
        return result
    
    def convert_w(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        datetime_admission = datetime(adm_y, adm_m, adm_d)
        admission_wkd = datetime_admission.weekday+1 # 默认周一为0， 周日为6
        v_w = int(matched)
        result = f'{admission_wkd - v_w}天'
        return result

    def convert_d(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_d = int(matched)
        result = f'{adm_d-v_d}天'
        return result

    def convert_ym(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_y, v_m = matched
        if v_y == '去':
            v_y = adm_y - 1
        elif v_y == '前':
            v_y = adm_y - 2
        return self.convert_ymd((str(v_y), v_m, '1'))
    
    def convert_ymd(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_y, v_m, v_d = matched
        datetime_admission = datetime(adm_y, adm_m, adm_d)
        datetime_v = datetime(int(v_y), int(v_m), int(v_d))
        delta = datetime_admission - datetime_v
        y_nums = delta.days//365
        m_nums = (delta.days % 365) // 30
        d_nums = delta.days
        if d_nums < 0:
            result = None
        elif y_nums > 0:
            result = f'{y_nums}年{m_nums}个月'
        elif m_nums > 0:
            result = f'{m_nums}个月'
        elif d_nums > 0:
            result = f'{d_nums}天'
        else:
            result = None
        return result
    
    def convert_md(self, matched):
        adm_y, adm_m, adm_d = self.admission_date
        v_m, v_d = matched
        v_y = adm_y
        datetime_admission = datetime(adm_y, adm_m, adm_d)
        datetime_v = datetime(int(v_y), int(v_m), int(v_d))
        delta = datetime_admission - datetime_v
        m_nums = delta.days // 30
        d_nums = delta.days % 30
        if m_nums > 0 and d_nums > 0:
            result = f'{m_nums}月{d_nums}天'
        elif m_nums > 0:
            result = f'{m_nums}月'
        elif d_nums > 0:
            result = f'{d_nums}天'
        else:
            result = None
        return result
    
    def convert_m_nums(self, matched):
        return f'{matched}个月'

    def convert_y_nums(self, matched):
        return f'{matched}年'
    
    def convert_special(self, matched):
        if re.search('年初', matched):
            v_m = 2
            adm_y, adm_m, adm_d = self.admission_date
            result = f'{adm_m-v_m}个月'
        else:
            result = None
        return result

    def get_regex_pipeline(self,):
        pipes = [
            (r'\d+\-\d+', None),
            (r'(\d{4})年(\d{1,2})月(\d{1,2})[日号]', self.convert_ymd),
            (r'(\d{4})年(\d{1,2})月', self.convert_ym),
            (r'([去前])年(\d{1,2})月', self.convert_ym),
            (r'(\d{1,2})月(\d{1,2})[日号]', self.convert_md),
            (r'(\d{1,2})月(?:份)?', self.convert_m),
            (r'(\d{1,2})个月', self.convert_m_nums),
            (r'(\d{1,2})年前', self.convert_y_nums),
            (r'(\d{4})年', self.convert_y),
            (r'(\d{1,2})周', self.convert_w_nums),
            (r'(\d{1,2})年', self.convert_y_nums),
            (r'(\d{1,2})[号日]', self.convert_d),
            (r'昨[天日]', '1天'),
            (r'今[天日]', '1天'),
            (r'前[天日]', '2天'),
            (r'去年', '1年'),
            (r'前年', '2年'),
            (r'上(个)?月', '1月'),
            (r'年初', self.convert_special),
            (r'上周(.+?)', '约1周'),
            (r'^上周$', '1周'),
            (r'周(.+?)', self.convert_w),
        ]
        pipes = [(re.compile(p), v) for p,v in pipes]
        return pipes
    def convert(self, v, admission_date):
        if v =='2年':
            a=1
        adm_y,adm_m,adm_d = re.findall('(\d{4})年(\d{1,2})月(\d{1,2})[日号]?', admission_date)[0]
        self.admission_date = (int(adm_y),int(adm_m),int(adm_d))
        for pat, handle_func in self.reg_pipes:
            matched = re.findall(pat, v)
            if not matched:
                continue
            if isinstance(handle_func, str):
                v = handle_func
            elif handle_func is None:
                v = v
            else:
                v = handle_func(matched[0])
            break
        return v

class TimeKeyFix:
    def __init__(self) -> None:
        self.time_v_converter = TimeValueConvert()
        pass
    def time_key_uniform(self, k, v, admission_date):
        if re.search(r'现病史.时间\d+.持续时间', k):
            k = k.replace('持续时间', '发生时间')
        elif re.search(r'现病史.时间\d+.发生时间',k):
            v = self.time_v_converter.convert(v, admission_date)
        if v is None:
            return ''
        else:
            return f'{k}:{v}'


if __name__ == '__main__':
    from merge_all_summary import ReOrderSummary
    from merge_all_summary import Method
    time_key_fix = TimeKeyFix()
    app = ReOrderSummary(
        merge_regular='inference/all_summary_keys/merge_regular.tsv',
        key_positions='inference/all_summary_keys/key_position.txt',
        similary_method=Method.fuzzywuzzy, # 
        regex_file = 'inference/all_summary_keys/regular_match.txt'
    )
    file=r'/data/hujunchao/record_gen/gpt4_continue_gen_new/qwen_histrounds15_filteroutthreash0.08_1批_2批0925_2批0926_时间问题重标注/test_common_streamNone_tmp.xlsx'
    result = []
    for sheet in ['Sheet1']:
        df = pd.read_excel(file, sheet_name=sheet)
        df['pred_output'] = df['pred_output'].fillna('')
        for idx, row in tqdm(df.iterrows()):
            row = dict(row)
            admission_date = row['admission_date']
            pred_output = row['pred_output']
            abs_list = app.split_v2(pred_output)
            abs_list_new = []
            for abs in abs_list:
                if not abs.strip() or not re.search(r':|：', abs):
                    continue
                k, v = re.split(r':|：', abs, maxsplit=1)
                new_abs = time_key_fix.time_key_uniform(k, v, admission_date)
                abs_list_new.append(new_abs)
            row['pred_output_fix'] = '\n'.join(abs_list_new)
            result.append({**row})
    result = pd.DataFrame.from_dict(result)
    result.to_excel(file.replace('.xlsx', '_timefix.xlsx'), index=False)