import re
from collections import OrderedDict
import traceback
import copy
from gensim.models import KeyedVectors
from enum import Enum
import jieba
import numpy as np
from tqdm import tqdm
import requests
import json
import sys
import os
from scipy.special import softmax
from scipy.spatial import distance
import pandas as pd
import jieba
import Levenshtein
from time_key_uniform import TimeKeyFix
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init_log import logger

class Method(Enum):
    fuzzywuzzy = 1
    gensim = 2

class SimilarVecUtil:
    def __init__(self, gensim_model_path=None, total_keys=None) -> None:
        self.gensim_model_path = gensim_model_path
        if self.gensim_model_path:
            self.word_vec = self.load_gensim_model()
        
        self.total_keys = total_keys
        self.key_vecs = self.cal_vec_for_each_key()
        pass
    
    def cal_vec_for_each_key(self):
        result = []
        for k in self.total_keys:
            result.append(self.cal_vec_for_key(k))
        return np.stack(result)

    def load_gensim_model(self):
        logger.info('loading gensim key word vector......')
        word_vec:KeyedVectors = KeyedVectors.load_word2vec_format(self.gensim_model_path, binary = False, encoding = 'utf-8', unicode_errors = 'ignore')
        for k in tqdm(word_vec.key_to_index.keys(), desc='token_add_to_jieba'):
            jieba.add_word(k)
        print('loading finished')
        return word_vec
    
    def get_vec_for_field(self, field):
        field_toks = jieba.lcut(field)
        field_vecs = np.stack([self.word_vec.get_vector(tok) for tok in field_toks])
        return np.mean(field_vecs, axis=0)
    
    def cal_vec_for_key(self, k):
        k_fileds = k.split('.')
        k_fileds_vec = [self.get_vec_for_field(field) for field in k_fileds]
        # fileds_proportion = softmax(range(len(k_fileds)))
        # proportion_vecs = np.expand_dims(fileds_proportion, 1)*k_fileds_vec
        return np.mean(k_fileds_vec, axis=0)

    def gensim_similar(self,k):
        vec = self.cal_vec_for_key(k)
        all_dist = distance.cdist(np.expand_dims(vec, 0), self.key_vecs)
        most_similar_idx = int(np.min(all_dist))
        return self.total_keys[most_similar_idx]

class ReOrderSummary:
    def __init__(self, merge_regular, key_positions, gensim_model_path=None,similary_method=None, regex_file=None) -> None:
        with open(merge_regular, encoding='utf8') as f:
            self.key_merge_regular = {line.split('\t')[0]:line.split('\t')[1].strip() for line in f.readlines()}
        ordered_keys = OrderedDict()
        with open(key_positions, 'r', encoding='utf-8')  as f:
            for line in f.readlines():
                ordered_keys[line.strip()] = ''
        self.ordered_keys = ordered_keys
        if similary_method == Method.gensim:
            self.similar_util = SimilarVecUtil(
                gensim_model_path=gensim_model_path, 
                total_keys=list(self.ordered_keys.keys())
            )
        else:
            self.similar_util = None
        self.similary_method = similary_method
        self.key_fields_layer, self.key_words = self.get_key_fields()
        self.regex_rule = self.load_regex_conf(regex_file)
        self.state = {'total_keys_num':0, 'no_norm_keys_num':0}
        self.time_key_fixer = TimeKeyFix()

    def load_regex_conf(self, regex_file):
        if not regex_file:
            return None
        result = []
        with open(regex_file, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#'):
                    continue
                line = f'(r{line})'
                line = eval(line)
                pattern = re.compile(line[0])
                if line[1] is not None:
                    replacement = line[1:]
                else:
                    replacement = None
                result.append((pattern, replacement))
        return result

    def get_key_fields(self):
        key_fields_layer = set()
        all_words = set()
        for k in self.ordered_keys.keys():
            k1 = k.split('.')[0]
            jieba.add_word(k1)
            jieba.add_word(k)
            key_fields_layer.add(k1)
            key_fields_layer.add(k)
            all_words.update(k)
        all_words = all_words - set('.') - set(str(i) for i in range(10))
        return key_fields_layer, all_words
    
    def regex_process_key(self, k:str):
        '''
        return None, delete, new_k
        '''
        def _fix_key(k:str, pat, replacements):
            matched_lst = re.findall(pat, k)
            if not isinstance(matched_lst[0], tuple):
                matched_lst = (matched_lst[0],)
            else:
                matched_lst = matched_lst[0]
            for i, matched in enumerate(matched_lst):
                k = k.replace(matched, replacements[i], 1)
            return k
        new_k = None
        for tp in self.regex_rule:
            # 匹配到的键，一律规则处理，否则不予处理，由后续流程继续处理
            if tp[0].search(k):
                if tp[1] is None:
                    new_k = 'delete'
                else:
                    new_k = _fix_key(k, tp[0], tp[1])
                    return new_k
        return new_k

    def fuzzywuzzy_similar(self, k):
        new_k = self.regex_process_key(k)
        if new_k == 'delete':
            return ''
        elif new_k is not None:
            return new_k
        elif len(set(k).intersection(self.key_words))/len(k) < 0.5:
            return ''
        else:
            all_std_keys = [k for k,v in self.ordered_keys.items()]
            result = sorted([(Levenshtein.distance(k, i), i) for i in all_std_keys], key=lambda x:x[0])
            return result[0][1]

    def convert_to_std_key(self, k):
        if self.similary_method == Method.fuzzywuzzy:
            return self.fuzzywuzzy_similar(k)
        elif self.similary_method == Method.gensim:
            return self.similar_util.gensim_similar(k)

    def split_v2(self, abs):
        abs_lst = jieba.lcut(abs)
        result = ['']
        for item in abs_lst:
            if item in self.key_fields_layer and len(re.split(r':|：', result[-1], maxsplit=1)) == 2 \
                    and len(re.split(r':|：', result[-1], maxsplit=1)[-1]) >0 :
                result.append(item)
            elif len(result[-1])>0 and result[-1][-1]=='\n':
                result.append(item)
            else:
                result[-1] += item
        result = [i.strip() for i in result if i.strip()]
        return result

    def uniform_kv_summary(self, summary, admission_date):
        try:
            ordered_keys = copy.deepcopy(self.ordered_keys)
            abs_list = self.split_v2(summary)
            self.state['total_keys_num'] += len(abs_list)
            result = []
            for abs in abs_list:
                if not abs.strip() or not re.search(r':|：', abs):
                    continue
                k, v = re.split(r':|：', abs, maxsplit=1)
                # 控制value长度
                v = str(v)[:50]
                if k not in ordered_keys:
                    self.state['no_norm_keys_num'] += 1
                    k_old = k
                    # 体格检查.体重:70kg
                    # 体格检查.血压:80/120mmhg
                    # 体格检查.血氧饱和度:95%
                    # 体格检查.巴氏征:阳性
                    # 体格检查.膝跳反应:阴性
                    if re.search(r'体格检查', k) and k.split('.')[1] not in ['体重', '身高']:
                        k = '体格检查.专科检查'
                        v = k_old.replace('体格检查.', '')+v
                    else:
                        k = self.convert_to_std_key(k_old)
                    logger.info(f'键匹配1：{k_old} -> {k}')
                if k:
                    try:
                        new_kv = self.time_key_fixer.time_key_uniform(k, v, admission_date)
                        # new_kv = f'{k}:{v}'
                    except Exception:
                        new_kv = f'{k}:{v}'
                    result.append(new_kv)
            new_summary = '\n'.join(result)
        except Exception:
            traceback.print_exc()
            new_summary = summary
        return new_summary

    def post_process_abs(self, all_abstract):
        """
        摘要后处理，对齐过滤
        1.按照给定的key序列排序；
        2.去重：根据key-value去重；根据子串去重合并；
        3.对于时间保留最后一个，前面删除；对于主要症状保留第一个，其他的插入作为伴随症状
        :param all_abstract:
        :return:
        """
        try:
            ordered_keys = copy.deepcopy(self.ordered_keys)
            conflict_main_symptom_keys = []
            time_to_accompany_order = {}
            abs_list = all_abstract.split("\n")
            for abs in abs_list:
                if not abs.strip() or not re.search(r':|：', abs):
                    continue
                k, v = re.split(r':|：', abs, maxsplit=1)
                if k not in ordered_keys:
                    # 待优化
                    # print('删除键：', abs)
                    # continue
                    k_old = k
                    k = self.convert_to_std_key(k_old)
                    logger.info(f'键匹配2：{k_old} -> {k}')
                # 取{字符串_S, 字符串_M}两种值，S代表单值，后边覆盖前面，M代表多值，要合并
                merge_type = self.key_merge_regular.get(re.sub(r'\d+', '1', k), '')
                is_main_symptom = '主要症状.症状术语' in k
                is_not_main_symptom = not is_main_symptom

                # 主要症状处理逻辑
                if is_main_symptom:
                    if not ordered_keys[k]:
                        ordered_keys[k] = v
                    else:
                        # 冲突的主要症状要转换为伴随症状，需考虑序号，先暂存起来
                        conflict_main_symptom_keys.append((k,v))
                    continue
                # 其它键处理逻辑
                else:
                    if merge_type.endswith('_S'):
                        ordered_keys[k] = v
                    elif merge_type.endswith('_M'):
                        if v not in ordered_keys[k] and re.sub(',|，', '、', v) not in ordered_keys[k]:
                            ordered_keys[k] = f'{ordered_keys[k]}，{v}'
                    else:
                        logger.info(f'无效key：{k}')
                    # 需获取伴随症状序号，用以主要症状的键的更新
                    if '伴随症状' in k:
                        k_fields = k.split('.')
                        time_order = '.'.join(k_fields[:2])
                        last_order = time_to_accompany_order.get(time_order, 0)
                        time_to_accompany_order[time_order] = max(int(k_fields[2][-1]), last_order)
            for confict_k, v in conflict_main_symptom_keys:
                time_order = '.'.join(confict_k.split('.')[:2])
                new_accompany_order = time_to_accompany_order.get(time_order, 0) + 1
                new_key = confict_k.replace('主要症状', f'伴随症状{new_accompany_order}')
                time_to_accompany_order[time_order] = new_accompany_order
                ordered_keys[new_key] = v

            result = [f'{k}:{v.strip(",，")}' for k,v in ordered_keys.items() if v.strip()]
        except:
            traceback.print_exc()
            return all_abstract
        return "\n".join(result)

def test_on_unseen_data(app:ReOrderSummary):
    root = r'E:\data\病历生成\持续生成_gpt4_new\持续生成训练集\待标注数据-20231024-预标\baichuan_预标2'
    result = []
    for excel in tqdm(os.listdir(root)):
        df = pd.read_excel(os.path.join(root, excel)).fillna('')
        for idx, summ in enumerate(df['过程摘要_预标']):
            if summ:
                summ_new = app.uniform_kv_summary(summ)
                if summ_new != summ:
                    result.append({'file':excel, 'line':idx, 'origin':summ, 'fixed':summ_new})
    result = pd.DataFrame.from_dict(result)
    result.to_excel(root+'_键标准化问题.xlsx')
    logger.info(f'key state:{app.state}')

def summary_request(dia_id, dia):
    header = {
        'Content-Type': 'application/json',
    }
    url = 'http://10.128.3.145:7915/gen_abstract_and_record_v1'

    req_data = {
        'model_type':'shanhai',
        'diag_id':dia_id,
        'diag':dia,
        'diag_turn_num':1
    }
    ret = requests.post(url=url, headers=header, data=json.dumps(req_data))
    res = json.loads(ret.content.decode('utf-8'))['data']
    return res

def report_request(dia_id, abs):
    url='http://10.200.8.38:7915/gen_record_by_abstract_v1'
    header = {
        'Content-Type': 'application/json',
    }
    abs = '\n'.join([f'{k}:{v}' for k,v in abs.items()])
    req_data={
        'model_type':'shanhai',
        'diag_id':dia_id,
        'abstract':abs,
    }
    ret = requests.post(url=url, headers=header, data=json.dumps(req_data))
    res = json.loads(ret.content.decode('utf-8'))['data']
    return res

def cot_verify(file=None):
    if file is None:
        file = r'C:\Users\YZS\Desktop\train_correct_asr_shanhai_hasinfo_abtract_turnNum\result_test_abstract_1113_477_conversation_姜磊(1).xlsx'
        file=r'20231017新增_兼职_复核30份_重复核/04876462_1_asr_内分泌科_None.xlsx'

    df = pd.read_excel(file, dtype=str)
    result = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        i = row['当前对话']
        record_name = 'hjc_'+row['record_id']
        res = summary_request(record_name, i)
        # res = '\n'.join([f'{k}:{v}' for k,v in res.items()])
        row['tmp'] = res
        if res:
            result.update(res)
    # pd.DataFrame.from_dict(result).astype(str).to_excel('')
    result
    report = report_request(record_name, result)
    return report, result

if __name__ == '__main__':
    all_abstract = r'''
现病史.时间1.发生时间:2天
现病史.时间1.主要症状.症状术语:急性胰腺炎
初步诊断.疾病名称1:急性胰腺炎
现病史.时间1.诊治经过.医疗机构名称:保定当地医院
现病史.时间1.病因与诱因:高血脂
现病史.时间1.诊治经过.病情转归:2022年2月、11月因急性胰腺炎再次住院治疗
现病史.时间1.诊治经过.病情转归:2022年11月因急性胰腺炎住院治疗
处理意见.检查项目名称:胰腺炎诱因检查
现病史.时间1.诊治经过.病情转归:已控制饮食，降血脂治疗
处理意见.其他建议:注意饮食
现病史.时间1.发生时间:2年
现病史.时间1.主要症状.症状术语:急性胰腺炎
初步诊断.疾病名称1:急性胰腺炎
现病史.时间1.发生时间:2年
现病史.时间1.诊治经过.病情转归:2022年2月、11月因急性胰腺炎再次住院治疗
现病史.时间1.发生时间:1月
现病史.时间1.诊治经过.检查检验项目及结果:淀粉酶升高3倍以上
现病史.时间1.主要症状.症状术语:腹痛
现病史.时间1.诊治经过.检查检验项目及结果:血脂、淀粉酶升高
处理意见.检查项目名称:血脂
处理意见.检查项目名称:大生化
现病史.时间1.诊治经过.检查检验项目及结果:甘油三酯升高
现病史.时间1.诊治经过.检查检验项目及结果:甘油三酯最高达11mmol/L
处理意见.其他建议:注意饮食
处理意见.其他建议:半个月之后复查
处理意见.其他建议:三个月之后复查
处理意见.其他建议:三个月之后复查
处理意见.其他建议:间隔时间较长时，血脂会偏高，需控制饮食
现病史.时间.时间1.主要症状.症状术语:腹痛
现病史.时间2.主要症状.症状术语:进食烧烤后突发腹痛
现病史.时间1.诊治经过.检查检验项目及结果:CT示胰腺肿大、渗出
现病史.时间1.诊治经过.检查检验项目及结果:血脂、淀粉酶升高，甘油三酯最高达11mmol/L
现病史.时间.时间1.诊治经过.检查检验项目及结果:乳糜血
初步诊断.疾病名称2:高脂血症
处理意见.其他建议:高脂血症建议内分泌科就诊
现病史.时间.时间1.诊治经过.病情转归:已控制饮食，疼痛缓解后出院
现病史.时间.时间.时间1.诊治经过.检查检验项目及结果:血脂、淀粉酶升高
现病史.时间.时间1.诊治经过.检查检验项目及结果:血脂、淀粉酶升高
既往史.其他信息:无其他基础疾病
既往史.其他信息:孕期突发急性胰腺炎
既往史.疾病史1.疾病名称:脂肪肝
既往史.其他信息:无长期用药史
既往史.疾病史.其他信息:发病后曾口服他汀类药物
既往史.其他信息.药物过敏:消炎药
既往史.其他信息.药物过敏:青霉素
处理意见.检查项目名称:消化系统检查
处理意见.其他建议:高脂血症建议内分泌科就诊
初步诊断.疾病名称2:高脂血症
初步诊断.疾病名称3:慢性胰腺炎
处理意见.其他建议:高脂血症建议内分泌科就诊
现病史.时间1.发生时间:9天
处理意见.其他建议:预约超声内镜
处理意见.检查项目名称:超声内镜
处理意见.检查项目名称:胃结石
现病史.时间.时间1.诊治经过.检查检验项目及结果:CT示胆囊泥沙样结石
初步诊断.疾病名称4:胆囊泥沙样结石
初步诊断.疾病名称2:高脂血症
处理意见.其他建议:高脂血症建议内分泌科就诊
初步诊断.疾病名称4:胆囊泥沙样结石
处理意见.检查项目名称:血脂
体格检查.体重:70kg
体格检查.血压:80/120mmhg
体格检查.血氧饱和度:95%
体格检查.巴氏征:阳性
体格检查.膝跳反应:阴性
    '''
    all_abstract2 = r'''
现病史.时间1.主要症状.特点:aaa
体格检查.体重:70kg
体格检查.血压:80/120mmhg
体格检查.血氧饱和度:95%
体格检查.巴氏征:阳性
体格检查.膝跳反应:阴性
现病史.时间1.阴性症状:无心慌，无心慌，现病史.时间1.伴随症状2.症状术语，怕热
'''
    # with open(os.path.join(PROJECT_ROOT, 'scripts', 'training', 'all_summary_keys', 'merge_summary_logs_sorted.txt'), encoding='utf8') as f:
    #     lines = '\n'.join(i.split(' ->')[0] for i in f.readlines())
    #     all_abstract2 = lines

    # app = ReOrderSummary(
    #     merge_regular=os.path.join(PROJECT_ROOT, 'scripts', 'training', 'all_summary_keys', 'merge_regular.tsv'),
    #     key_positions=os.path.join(PROJECT_ROOT, 'scripts', 'training', 'all_summary_keys', 'key_position.txt'),
    #     gensim_model_path=r'E:\bert_models\chinese_word_vector\sgns.baidubaike.bigram-char.bz2', # Method.gensim 时有效
    #     similary_method=Method.fuzzywuzzy, #
    #     regex_file = os.path.join(PROJECT_ROOT, 'scripts', 'training', 'all_summary_keys', 'regular_match.txt')
    # )
    app = ReOrderSummary(
        merge_regular=r'inference/all_summary_keys/merge_regular.tsv',
        key_positions=r'inference/all_summary_keys/key_position.txt',
        regex_file=r'inference/all_summary_keys/regular_match.txt',
        # gensim_model_path=r'sgns.baidubaike.bigram-char.bz2',  # Method.gensim 时有效
        similary_method=Method.fuzzywuzzy  #
    )

    # print(app.post_process_abs(all_abstract))
    # print(app.uniform_kv_summary('现病史.时间1.诊治经过.检查检验项目及结果:超声提示颈动脉斑块、下肢动脉硬化，初步诊断.疾病名称3:颈动脉硬化，初步诊断.疾病名称4:冠心病，初步诊断.疾病名称5:高脂血症', '2023年8月30'))
    # cot_verify()
    result = []
    for file in os.listdir('20231017新增_兼职_复核30份_重复核'):
        path=os.path.join('20231017新增_兼职_复核30份_重复核', file)
        record_id = file.split('_')[0]
        try:
            report, summary = cot_verify(file=path)
            result.append((record_id, report, summary))
        except Exception:
            pass
        
    pd.DataFrame.from_records(result, columns=['record_id', 'report', 'summary']).to_excel('tmp.xlsx')