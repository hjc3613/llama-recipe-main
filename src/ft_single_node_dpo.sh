# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FT_MODEL_TYPE='qwdpo'
base_model=dpo_14b_to_72b_spin_iter6-Qwen-14B-Base
model_name=/fl-ift/med/hujunchao/models/${base_model}
model_name=${model_name%/} # 删除结尾的/
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/DPO/
file_name=dpo_14b_to_72b_spin_iter7.xlsx
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}
# python -m pdb \
torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning_dpo.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --low_cpu_fsdp \
    --enable_fsdp \
    --batch_size_training 2 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 1 \
    --gradient_accumulation_steps 2 \
    --fsdp_config.pure_bf16 \
    --lr 1e-6 \
    --seed 0 \
    --gradient_clipping \
    --gradient_clipping_threshold 5 \
    # --warmup_steps 150 \
    # --optimizer RMSprop \
    # --fsdp_cpu_offload \
    # --optimizer PagedAdamW32bit \
    
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/} \
    --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/} \

cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}/