# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
MODE=$1
echo 'MODE: '$MODE
# export CUDA_VISIBLE_DEVICES=2
export FT_MODEL_TYPE='qw2'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
# model_name='/fl-ift/med/common/Qwen-72B-Chat'
model_name='/fl-ift/med/common/Qwen2-72B-Instruct'
# model_name='/fl-ift/med/common/Qwen-14B-Chat'
# model_name='/fl-ift/med/common/llama3-openbiollm-8b'
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report
file_name='胸部_腹部_头颅'
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}

if [[ $MODE == *"train"* ]]; then 
    torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning.py \
    --model_name ${model_name} \
    --dataset_format 'text' \
    --dataset ${dataset} \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 5 \
    --gradient_accumulation_steps 4 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 2e-5 \
    --seed 123456 \
    --gradient_clipping \
    --context_length 4096 \
    --save_checkpoint_every_step 0 \
    --warmup_ratio 0.3 \
    --weight_decay 0.1 \
    --scheduler cosine \
    --update_lr_every_step 1 \
    --freeze_layers \
    --num_freeze_layers 40 \
    --freeze_strategy 1 \
    --parallel_granularity 'Qwen2SdpaAttention-Qwen2FlashAttention2-Qwen2Attention-Qwen2MLP-Qwen2RMSNorm' `# weight、decoder_layer、QWenAttention-QWenMLP-RMSNorm、QWen2SdpaAttention-QWen2FlashAttention2-QWen2Attention-QWen2MLP-QWen2RMSNorm` \
    # --fsdp_cpu_offload \
    # --max_train_step 100 \
    # --gamma 0.6 \
    # --optimizer PagedAdamW32bit \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO/test${idx}.xlsx \
    # --save_metrics \

fi
    
if [[ $MODE == *"merge"* ]]; then 
    python convert_fsdp_to_hf.py \
        --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/}-stepfinal \
        --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal
fi

if [[ $MODE == *"copy"* ]]; then 
    cp ${model_name}/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal/
fi