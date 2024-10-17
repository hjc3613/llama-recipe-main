# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
export FT_MODEL_TYPE='qw2'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
model_name='/fl-ift/med/common/Qwen2-72B-Instruct'
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2
file_name=all_part_diagnose_fix_xuhao
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}
master_ip='10.233.71.52'
torchrun --nnodes 2 --nproc_per_node 8 --rdzv-id=7788 --rdzv-backend=c10d --rdzv-endpoint=${master_ip} \
    finetuning.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --dataset_format 'text' \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 2 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 1 \
    --gradient_accumulation_steps 1 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 2e-4 \
    --seed 123456 \
    --gradient_clipping \
    --context_length 4096 \
    --save_checkpoint_every_step 100 \
    --update_lr_every_step 1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --scheduler cosine \
    --parallel_granularity 'weight' \
    # --freeze_layers \
    # --freeze_strategy 6 \
    # --gamma 1 \
    # --fsdp_cpu_offload \
    # --optimizer PagedAdamW32bit \
    # --num_freeze_layers 45 \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO/test${idx}.xlsx \
    
    
IP=`hostname -I`
if [ $IP == ${master_ip} ]
then
    time python convert_fsdp_to_hf.py \
        --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/} \
        --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/} \

    # cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}/

fi
