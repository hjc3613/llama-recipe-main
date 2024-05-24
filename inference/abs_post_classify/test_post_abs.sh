data_dir=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/dialogue2abs_20240416
result_dir=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/dialogue2abs_20240416
#infile=test_1110_2725_conversation_post_class_biaozhu_1610.jsonl
#infile=test_special_1211_86.jsonl
#infile=test_0128.jsonl
#infile=复诊一致性测试集.jsonl
infile=复诊线上日志.jsonl
#hf_home=/kanas/nlp/lihui/mymodels/abs_post_pretrain1210_0204
#hf_home=/kanas/nlp/lihui/mymodels/classify_dialog2abs_report2abs_0222
#hf_home=/kanas/nlp/lihui/mymodels/MFTCoder/final_step_585
#hf_home=/kanas/nlp/lihui/mymodels/classify_dialog2abs_report2abs_selfpaced_0308/checkpoint-602
#hf_home=/kanas/nlp/lihui/mymodels/classify_dialog2abs_report2abs_weigthloss_0311/
#hf_home=/kanas/nlp/lihui/mymodels/MFTCoder/step_100/
#hf_home=/kanas/nlp/lihui/mymodels/abs_post_qwen1.5/
#hf_home=/kanas/nlp/lihui/mymodels/abs_post_ensemble
#hf_home=/baykal/unisound/jianglei/outputs/outputs/abs_models/qwen_1212_classify_add1637cls_seed12345
hf_home=/fl-ift/med/hujunchao/models/train_classify_3106_1637_0319-UniGPT2-08-01-00-240315
python cascade_decode_abs_classify.py $data_dir/$infile 2 $result_dir $hf_home 2 base
