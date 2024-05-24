# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FT_MODEL_TYPE='qwdpo'
# base_model=Qwen-72B
# model_name=/fl-ift/med/common/${base_model}
model_name=/fl-ift/med/jianglei/project/llama-recipes-main/src/checkpoints_dia2abstract2record_0410_348x2_base_17key_claer_72B_35layer_epoch6_b8_2e5_gc0_wd0_seed12345/hf_3
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/dialogue2record_orpo
file_name=orpo_train.xlsx
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}
torchrun --nnodes=2 --nproc_per_node=8 --rdzv-id=7789 --rdzv-backend=c10d --rdzv-endpoint=10.233.96.188 \
    finetuning_dpo.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --low_cpu_fsdp \
    --enable_fsdp \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 3 \
    --gradient_accumulation_steps 4 \
    --fsdp_config.pure_bf16 \
    --lr 1e-6 \
    --seed 0 \
    --gradient_clipping \
    --gradient_clipping_threshold 5 \
    # --fsdp_cpu_offload \
    # --optimizer PagedAdamW32bit \
    
