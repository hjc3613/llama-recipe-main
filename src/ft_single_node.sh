# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
idx=$1
export FT_MODEL_TYPE='qw'

model_name='/fl-ift/med/common/Qwen-14B-Base'
# model_name='/fl-ift/med/hujunchao/models/mixtral-8x7b-instruction-v0.1'
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/common/datasets/med/mix_med_common/version01
file_name=chunked_bin/dataset_packed
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}

torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning.py \
    --model_name ${model_name} \
    --dataset_format 'bin' \
    --dataset ${dataset} \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 4 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 1 \
    --gradient_accumulation_steps 16 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 1e-5 \
    --seed 123456 \
    --gradient_clipping \
    --context_length 2500 \
    --save_checkpoint_every_step 200 \
    --update_lr_every_step 1 \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --scheduler constant_with_warmup \
    --save_metrics \
    # --freeze_layers \
    # --freeze_strategy 6 \
    # --max_train_step 100 \
    # --gamma 0.6 \
    # --num_freeze_layers 48 \
    # --fsdp_cpu_offload \
    # --optimizer PagedAdamW32bit \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO/test${idx}.xlsx \
    
    
    
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/}-stepfinal \
    --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal \

cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal/
