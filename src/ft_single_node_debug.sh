# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
idx=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FT_MODEL_TYPE='qw'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
model_name='/fl-ift/med/common/Qwen-14B-Base'
# model_name='/fl-ift/med/hujunchao/models/mixtral-8x7b-instruction-v0.1'
# model_name='/fl-ift/med/common/Qwen-14B-Chat'
# model_name='/fl-ift/med/common/llama3-openbiollm-8b'
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2
file_name='腹部_胸部_头颅_诊断'
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}

# torchrun --nnodes 1 --nproc_per_node 8 \
python -m pdb \
    finetuning.py \
    --model_name ${model_name} \
    --dataset_format 'text' \
    --dataset ${dataset} \
    --low_cpu_fsdp \
    --batch_size_training 4 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 5 \
    --gradient_accumulation_steps 1 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 2e-5 \
    --seed 123456 \
    --gradient_clipping \
    --context_length 4096 \
    --save_checkpoint_every_step 0 \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --scheduler cosine \
    --update_lr_every_step 1 \
    # --enable_fsdp \
    # --freeze_layers \
    # --num_freeze_layers 40 \
    # --freeze_strategy 1 \
    # --fsdp_cpu_offload \
    # --save_metrics \
    # --max_train_step 100 \
    # --gamma 0.6 \
    # --optimizer PagedAdamW32bit \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO/test${idx}.xlsx \
    
    
    
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/}-stepfinal \
    --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal \

cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}-stepfinal/
