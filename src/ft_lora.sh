# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO

torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning.py \
    --model_name /fl-ift/nlp/hujunchao/models/Qwen-14B-Base/ \
    --dataset /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/tmp/train_dialogue2abs_report2abs_635.xlsx \
    --use_peft \
    --enable_fsdp  \
    --fsdp_config.pure_bf16 \
    --peft_method lora \
    --batch_size_training 1 \
    --output_dir lora_ft \
    --num_epochs 5 \
    --gradient_accumulation_steps 8 \
    --batching_strategy padding \
    --lr 2.5e-5 \
    --seed 123456 \
    --gradient_clipping \
    # --run_validation \
    # --val_ds /fl-ift/nlp/hujunchao/git_root/llama-recipes-main/data/data_merge_d2a_r2a_xiaohua_110_coarse_grained/train_dialogue2abs_report2abs_635_val.xlsx \
    # --save_metrics \
    # --optimizer PagedAdamW32bit
    # --fsdp_cpu_offload \
