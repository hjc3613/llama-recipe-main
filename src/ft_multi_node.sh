# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
export FT_MODEL_TYPE='qw2'

model_name='/fl-ift/med/common/Qwen1.5-72B-Chat'
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report
file_name=train_abnormal_to_template_2_fix_ASR_right_2.xlsx
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}

torchrun --nnodes 2 --nproc_per_node 8 --rdzv-id=7788 --rdzv-backend=c10d --rdzv-endpoint=10.233.97.138 \
    finetuning.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 5 \
    --gradient_accumulation_steps 2 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 2e-5 \
    --seed 123456 \
    --gradient_clipping \
    --fsdp_cpu_offload \
    # --freeze_layers \
    # --num_freeze_layers 45 \
    # --freeze_strategy 1 \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO/test${idx}.xlsx \
    # --optimizer PagedAdamW32bit
    
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/} \
    --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/} \

# cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}/