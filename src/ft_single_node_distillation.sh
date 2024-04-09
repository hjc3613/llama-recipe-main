# export LD_LIBRARY_PATH=/data/hujunchao/miniconda/envs/py310/lib/:${LD_LIBRARY_PATH}
# export NCCL_DEBUG=INFO
idx=$1
export FT_MODEL_TYPE='qwdistillation'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
base_model=quant_test_model_0319_1250_abstract
model_name=/fl-ift/med/common/${base_model}/hf_pruning_25
data_root=/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/distillation
file_name=train_dialogue2report_20240226_368_new_adddata_0319_pres_abstract_front_367_1250.xlsx
dataset=${data_root}/${file_name}
dist_checkpoint_root_folder=checkpoints_qw
dist_checkpoint_folder=${file_name%.*}

# export DISTILLATION_MODE='eval'
# export SAVE_HIDDEN_STATES_BI='False'
# export SAVE_HIDDEN_STATES='True'
# mkdir -p distillation_hidden_states3
# # python \
# torchrun --nnodes 1 --nproc_per_node 8 \
#     finetuning_distillation.py \
#     --model_name ${model_name} \
#     --dataset ${dataset} \
#     --batch_size_training 1 \
#     --num_epochs 1 \
#     --batching_strategy padding \
#     --enable_fsdp \
#     --low_cpu_fsdp \
    
export DISTILLATION_MODE='train'
export SAVE_HIDDEN_STATES='False'
torchrun --nnodes 1 --nproc_per_node 8 \
    finetuning_distillation.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder ${dist_checkpoint_root_folder} \
    --dist_checkpoint_folder ${dist_checkpoint_folder} \
    --num_epochs 4 \
    --gradient_accumulation_steps 4 \
    --fsdp_config.pure_bf16 \
    --batching_strategy padding \
    --lr 2e-5 \
    --seed 123456 \
    --gradient_clipping \
# time python convert_fsdp_to_hf.py \
#     --fsdp_checkpoint_path  ${dist_checkpoint_root_folder}/${dist_checkpoint_folder}-${base_model} \
#     --consolidated_model_path /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${base_model} \

# cp /fl-ift/med/common/Qwen-14B-Base/*.py /fl-ift/med/hujunchao/models/${dist_checkpoint_folder}-${base_model}/