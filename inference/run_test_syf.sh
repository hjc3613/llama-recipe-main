export CUDA_VISIBLE_DEVICES=2,3,4,5,6
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

MODEL_ROOT='/fl-ift/med/hujunchao/models/'
model_path='mixed_task_radiology-unigpt_17b_5epoch_bf16'
model_path='mixed_task_radiology-unigpt_17b_5epoch'
model_path='mixed_task_radiology_noise-unigpt_17b_5epoch'
model_path='qwen1.5-110b-tmp'
# model_path='RadiologyReport_20240618_胸部_腹部_头颅_胸X_qwen'
# model_path='mixed_task_radiology_common_instruct2-unigpt_pro_17B'
# model_path=/fl-ift/med/common/Qwen1.5-14B-Chat
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/多任务/导出报告-20240426-端到端效果待评估-分析_prompt1_test.xlsx'
file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology_test/test_yingxiang/胸部CT_abnorm_test_preprocessed_prompt2.xlsx'
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology_test/test_yingxiang/腹部CT_abnorm_test_preprocessed_prompt2.xlsx'
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology_test/test_yingxiang/头颅CT_abnorm_test_preprocessed_prompt2.xlsx'
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology_test/test_yingxiang/胸部X片_abnorm_test_preprocessed_prompt2.xlsx'
# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/mixed_task_radiology_test/test_multi_template/end2end_multi_template_test_mix.xlsx'

# file='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/multi_template_chaoyang/放射科模板_异常_报告2.xlsx'
example_nums=1500
python infer_script.py \
    --model_path ${model_path} \
    --file ${file} \
    --decode_type iter \
    --record_nums ${example_nums} \
    --stream False \
    --input_output True \
    --suffix pred_70b_epoch5_noise_qw2.5_bf16 \
    --sheet Sheet1 \
    --model_root ${MODEL_ROOT} \
    # --excel_dir '/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/X片'
