# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
# python -m pdb \
torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning_hf.py \
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
    --freeze_layers \
    --num_freeze_layers 35 \
    --freeze_strategy 1 \
    --fsdp_cpu_offload \
    # --warmup_steps 150 \
    # --optimizer RMSprop \
    # --optimizer PagedAdamW32bit \
    
    
time python convert_fsdp_to_hf.py \
    --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${model_name##*/} \
    --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/} \

cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${model_name##*/}/