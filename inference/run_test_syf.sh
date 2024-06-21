export CUDA_VISIBLE_DEVICES=6
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# strategy=report2summary_3data_merge
# strategy=hefei_fangan
# root=/data/yafei
# model_path=${root}/models/${strategy}/checkpoint-56
# file=../datasets/record_gen/${strategy}/test_report2abs_chuzhen_fix_merge_zhusu_xianbingshi.xlsx

#model_path='/data/yafei/models/qwen_histrounds15_filteroutthreash0.08_fuhe_0925_0926_1007fuhe/checkpoint-135'
# file='/data/yafei/record_gen/gpt4_continue_gen_new/qwen_histrounds15_filteroutthreash0.08_语音识别校正后对话测试/test.jsonl'
# model_path='/data/yafei/models/report2summary_gpt4_local_zengqiang/checkpoint-35'
# model_path='/data/yafei/models/data_merge_d2a_r2a_xiaohua_110_coarse_grained_fixed/checkpoint-265'
model_path='/fl-ift/med/hujunchao/models/胸部_腹部_头颅_胸X_报告-unigpt_pro_17B-stepfinal'
# model_path=/fl-ift/med/common/Qwen1.5-14B-Chat
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/多任务/导出报告-20240426-端到端效果待评估-分析_prompt1_test.xlsx'
file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/腹部_胸部_头颅_胸X_test/头颅_abnorm_test_preprocessed_prompt2.xlsx'
example_nums=300
python test_chixu_shengcheng.py \
    --model_path ${model_path} \
    --file ${file} \
    --decode_type iter \
    --record_nums ${example_nums} \
    --stream False \
    --input_output True \
    --suffix pred \
    --sheet Sheet1 \
    # --excel_dir '/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/X片'
