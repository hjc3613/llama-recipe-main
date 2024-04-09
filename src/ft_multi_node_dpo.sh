# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 2 --nproc_per_node 8 --rdzv-id=7788 --rdzv-backend=c10d --rdzv-endpoint=10.233.95.251 \
    finetuning_dpo.py \
    --model_name '/fl-ift/nlp/hujunchao/models/diag_to_key' \
    --dataset /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/DPO_anthropic/train_diag2key_dpo.xlsx \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder checkpoints_qw \
    --dist_checkpoint_folder dpo_lihui_beta0.1 \
    --num_epochs 2 \
    --gradient_accumulation_steps 2 \
    --fsdp_config.pure_bf16 \
    --lr 2e-6 \
    --seed 123456 \
    --gradient_clipping \
    # --fsdp_cpu_offload \
    # --optimizer PagedAdamW32bit \
    
