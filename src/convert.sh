# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
idx=$1
export FT_MODEL_TYPE='mixtral'
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  checkpoints_qw/dpo_14b_to_72b-mixtral-8x7b-instruction-v0.1 \
    --consolidated_model_path /fl-ift/med/hujunchao/models/dpo_14b_to_72b-mixtral-8x7b-instruction-v0.1 \
    # --HF_model_path_or_name /fl-ift/med/hujunchao/models/Qwen-72B-chat \

cp /fl-ift/med/hujunchao/models/mixtral-8x7b-instruction-v0.1/*.py /fl-ift/med/hujunchao/models/dpo_14b_to_72b-mixtral-8x7b-instruction-v0.1/
# cp /data/yafei/models/Qwen-14B-Base/configuration_qwen.py checkpoints_qw/hf/
