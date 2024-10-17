import requests
import pandas as pd
from tqdm import tqdm
import json
import os
import re

def create_prompt(text):
    p = f'给定一个腹部CT放射报告检查所见:\n{text}\n要求抽取处其中的异常部分，按行分隔，正常所见不能输出'
    return p

def local_api(text):
    url = r'http://localhost:7711/chat/completions'
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
        model="llama3.1_405",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""},
            ]
    )
    print("Chat response:", chat_response)

def run_test(file):
    # url = r'http://10.10.20.12:4033/chat/completions' # 基于模板生成影像所见
    # url = r'http://10.10.20.40:2003/chat/completions' # 基于影像所见生成影像诊断
    url = r'http://10.233.121.251:7711/chat/completions'
    # url = r'http://10.10.20.18:4934/chat/completions'
    # file =  r"E:\data\病历生成\影像报告2\友谊头部+腹部\腹部_abnorm_test_preprocessed_prompt2.xlsx"
    # file = r"E:\data\病历生成\影像诊断\腹部CT放射报告-生成结论-test.xlsx"
    # file = r"E:\data\病历生成\影像诊断\腹部CT放射报告-生成结论-test.xlsx"
    df = pd.read_excel(file, sheet_name='Sheet1').iloc[:400]
    # df = pd.read_excel(r"E:\data\病历生成\影像报告2\多任务\导出报告-20240426-端到端效果待评估-分析_prompt1_test.xlsx")
    result = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        text = row['input']
        text = create_prompt(text.replace('请根据上述影像报告，生成诊断结论:', ''))
        # text = get_example_text()
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
        print('-------------input-----------')
        print(row['input'])
        print('-------------pred------------')
        print(res)
        print('-------------label-----------')
        print(row.get('output'))
        print('================================================')
        row['pred'] = res
        result.append(row)
    result = pd.DataFrame.from_dict(result)
    result.to_excel(file.replace('.xlsx', '_online_0627.xlsx'))

def run_test2():
    url = r'http://10.1.0.27:7803/chat/completions' # 基于模板生成影像所见
    
    file =  r"E:\data\病历生成\影像报告2\友谊头部+腹部\腹部_abnorm_test_preprocessed_prompt2.xlsx"
    # file = r"E:\data\病历生成\影像诊断\腹部CT放射报告-生成结论-test.xlsx"
    # file = r"E:\data\病历生成\影像诊断\其它2\X线片\肩关节平片 2024年6月3日_prompt2.xlsx"
    df = pd.read_excel(file, sheet_name='Sheet1')
    # df = pd.read_excel(r"E:\data\病历生成\影像报告2\多任务\导出报告-20240426-端到端效果待评估-分析_prompt1_test.xlsx")
    result = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        text = row['input']
        # text = get_example_text()
        data = {
            "model": "MRG_1023",
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "你是一个医学专家，且精通放射报告、X线报告的知识与规范，请你竭尽所能为下列问题提供帮助"},
                {"role": "user", "content": text}],
            "max_tokens": 500,
            "repetition_penalty": 1,
            "stream": False
        }
        headers = {'Content-Type':'application/json'}
        res = requests.post(url=url, json=data, headers=headers).text
        try:
            res = json.loads(res)
        except:
            print(res)
        res = res['choices'][0]['message']['content']
        print('-------------input-----------')
        print(row['input'])
        print('-------------pred------------')
        print(res)
        print('-------------label-----------')
        print(row.get('output'))
        print('================================================')
        row['pred'] = res
        result.append(row)
    result = pd.DataFrame.from_dict(result)
    result.to_excel(file.replace('.xlsx', '_online_tmp.xlsx'))

def main():
    root = r'E:\data\病历生成\影像诊断\test_yingxiang_zhenduan'
    excels = os.listdir(root)
    for excel in excels:
        path = os.path.join(root, excel)
        run_test(path)

def get_abnormal_cache(path):
    df = pd.read_excel(path, sheet_name='Sheet5')
    df['suojian_abnormal'] = ''
    cache = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        suojian = row['影像表现']
        suojian_split = re.split(r'\s+', suojian.strip())
        suojian_res = []
        for item in suojian_split:
            if item in cache:
                res = cache[item]
            else:
                p = '''
instruction:找出放射报告所见描述中的异常，并用<abnormal>、</abnormal>包起来

example1:
input:肝脏形态大小如常，轮廓规整，肝实质内可见多发类圆形低密度影，边界清晰，最大者大小约5.1×3.6cm，部分病变边缘及内部分隔可见钙化，肝顶部可见斑片状钙化影。
output:肝脏形态大小如常，轮廓规整，<abnormal>肝实质内可见多发类圆形低密度影，边界清晰，最大者大小约5.1×3.6cm，部分病变边缘及内部分隔可见钙化</abnormal>，<abnormal>肝顶部可见斑片状钙化影</abnormal>。

example2:
input:脾脏形态大小如常，内可见类圆形低密度。
output:<abnormal>脾脏形态大小如常，内可见类圆形低密度</abnormal>。

example3:
input:双肾增大，双肾可见多发类圆形低密度影，部分边缘钙化，双侧肾窦可见斑点状钙化密度影，双侧肾盂及输尿管未见扩张。
output:<abnormal>双肾增大，双肾可见多发类圆形低密度影，部分边缘钙化</abnormal>，<abnormal>双侧肾窦可见斑点状钙化密度影</abnormal>，双侧肾盂及输尿管未见扩张。

example4:
input:胆囊大小如常，壁无增厚，腔内未见异常密度影。肝内外胆管未见扩张。
output:胆囊大小如常，壁无增厚，腔内未见异常密度影。肝内外胆管未见扩张。

example5:
input:子宫大小形态如常，子宫前部可见结节样等密度突隆，双侧附件区未见异常密度影。膀胱充盈欠佳，腔内未见异常密度影。双侧盆壁及腹股沟区未见肿大淋巴结。
output:子宫大小形态如常，<abnormal>子宫前部可见结节样等密度突隆</abnormal>，双侧附件区未见异常密度影。<abnormal>膀胱充盈欠佳</abnormal>，腔内未见异常密度影。双侧盆壁及腹股沟区未见肿大淋巴结。

example6:
input:{item}
output:
'''
                p = p.format_map({'item':item})
                # _, res = ner_api_single(p, model='unisound#unigpt-medical-4.0')
                _, res = local_api(p)
                cache[item] = res
            suojian_res.append(res)
        suojian_res = '\n'.join(suojian_res)
        df.at[idx, 'suojian_abnormal'] = suojian_res
    df.to_excel(path.replace('.xlsx', '_suojian_abnormal.xlsx'))

def get_abnormal_each_part(path):
    df = pd.read_excel(path, sheet_name='Sheet1').iloc[:]
    df['suojian_lines2'] = ''
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # if row['clean_lines'] != 1:
        #     continue
        suojian = row['影像表现']
        p = f'给定影像学表现：\n{suojian}\n请按部位进行拆分，拆分部位必须在以下列表范围内：\n肝\n胆\n脾\n胰\n肾\n肾盂\n肾上腺\n子宫及附件区\n前列腺及精囊腺\n膀胱\n腹腔及腹膜后间隙\n盆壁及腹股沟\n胃肠道\n其它\n请对上述影像学表现进行拆分，将不同部位的影像学表现分别列出。输出json结构，key为部位，value为对应的影像学表现。'
        p = f'给定影像学表现：\n{suojian}\n请按部位进行拆分，拆分部位必须在以下列表范围内：\n肝\n胆\n脾\n胰\n肾\n肾盂\n肾上腺\n{"子宫及附件区" if "子宫" in suojian else "前列腺及精囊腺"}\n膀胱\n腹腔及腹膜后间隙\n盆壁及腹股沟\n胃肠道\n其它\n请对上述影像学表现进行断句拆分，将不同部位的影像学表现分别列出。输出顺序按原文中的顺序为准，每个部位单独写一行，拆分结果以<start>开始，以</end>结束，具体的，输出格式为：\n<start>部位1：部位1的所见原文\n部位2：部位2的所见原文\n部位3：部位3的所见原文\n...\n部位n：部位n的所见原文</end>，未提及的不要输出'
        p = f'给定影像学表现：\n{suojian}\n，请分别列出下列部位在影像学表现中的原文是什么：\n肝\n胆\n脾\n胰\n肾\n肾盂\n肾上腺\n{"子宫及附件区" if "子宫" in suojian else "前列腺及精囊腺"}\n膀胱\n腹腔及腹膜后间隙\n盆壁及腹股沟\n胃肠道\n其它部位\n输出格式为：\n<start>部位1：部位1的所见原文\n部位2：部位2的所见原文\n部位3：部位3的所见原文\n...\n部位n：部位n的所见原文</end>，如果某个部位被切分后语义不完整，请保留完整语义。'
        _, res = local_api(p)
        print(res)
        df.at[idx, 'suojian_lines2'] = res
    df.to_excel(path.replace('.xlsx', '_suojian_lines.xlsx'))

def add_number_to_diag():
    import dask.dataframe as dd
    import swifter

    def fix_diag(row):
        output = row['output']
        if output.startswith('1'):
            res = output
        else:
            p = f'''
instruction:为影像诊断中的每个诊断添加序号，若影像诊断中已有诊断序号，则返回源文本，否则返回添加过序号的的文本，返回结果以<start>开始，以<end>结束
举例1
input:胆囊壁增厚，炎症不除外，请结合临床；胰头、钩突区多发结节状致密影；食道下段管壁稍厚；阑尾腔粪石可能；右髋术后。
output:<start>1、胆囊壁增厚，炎症不除外，请结合临床；2、胰头、钩突区多发结节状致密影；3、食道下段管壁稍厚；4、阑尾腔粪石可能；5、右髋术后。<end>
举例2
input:前列腺钙化。
output:<start>1、前列腺钙化。<end>
举例3
input:1、左侧肱骨大结节骨折；2、左肩关节盂下缘形态似稍欠规则，请结合临床；3、左肩关节周围软组织肿胀；4、左肩关节退行性改变。
output:<start>1、左侧肱骨大结节骨折；2、左肩关节盂下缘形态似稍欠规则，请结合临床；3、左肩关节周围软组织肿胀；4、左肩关节退行性改变。<end>
参考上述样例，对下列影像诊断添加序号，逻辑同上
input:{output}
output:
'''
            _, res = local_api(p)
            res = res.replace('<start>', '').replace('<end>', '')
        return res
    root = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2/all_part_diagnose'
    os.makedirs(root+'_fix_xuhao', exist_ok=True)
    for excel in os.listdir(root):
        path = os.path.join(root, excel)
        df = pd.read_excel(path).iloc[:]
        # df = dd.from_pandas(df_org, npartitions=10)
        result = df.swifter.set_npartitions(1).apply(fix_diag, axis=1)
        # result = [i['output_new'] for i in result]
        df['output_old'] = df['output']
        df['output'] = result
        df.to_excel(os.path.join(root+'_fix_xuhao', excel), index=False)

def find_all_abnormals():
    root = r"/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/pred_test_0715_online"
    for excel in os.listdir(root):
        path = os.path.join(root, excel)
        df = pd.read_excel(path)
        df['异常'] = ''
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            biaoxian = row['input'].replace('请根据上述影像报告，生成诊断结论:', '')
            p = f'给定放射报告所见：\n{biaoxian}\n请抽取其中的所有异常表现，按部位划分，每个部位单独一行，每行以部位名字开头，无异常的部位不要出书任何信息，只针对有异常的部位输出其异常表现，最终的抽取结果以<start>开始，以<end>结束，若所有部位均无异常表现，则输出None，请输出抽取结果：'
            _, res = local_api(p)
            df.at[idx, '异常']=res.replace('<start>', '').replace('<end>', '')
        df.to_excel(path.replace('.xlsx', '_辅助信息.xlsx'))

def fangshe_test():
    file = r'/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/multi_template_chaoyang/放射科模板_异常_报告2.xlsx'
    df = pd.read_excel(file)
    df['pred'] = ''
    for idx, row in df.iterrows():
        input = row['input']
        _, res = local_api(input)
        print('--------------------------- input ---------------------------')
        print(input)
        print('--------------------------- output --------------------------')
        print(res)
        print('=============================================================')
        df.at[idx, 'pred'] = res
    df.to_excel(file.replace('.xlsx', 'epoch5_72b_quant.xlsx'))

if __name__ == '__main__':
    # run_test('/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论3/腹部CT放射报告-生成结论-trian3.xlsx')
    # main()
    # get_abnormal_each_part('/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/end2end_train/腹部2024年6月3日(1)_suojian_lines_cleaned_lines.xlsx')
    # add_number_to_diag()
    # find_all_abnormals()
    # local_vllm()
    fangshe_test()