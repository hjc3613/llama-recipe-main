# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=False
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    dataset_format: str='text'
    val_ds = None
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    freeze_reverse: bool = False # 默认为False，即只冻结特定层，激活其它层；若为True，则相反，只激活特定层，冻结其它层，为适配论文llama_pro
    num_freeze_layers: int = 1
    freeze_strategy: int = 1 # 冻结层的策略，自定义，比如等于3时，只冻结3、6、9......层，层数从1开始计算，而非0
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    scheduler: str = 'StepLR' # 'StepLR', or transformers scheduler(linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt, reduce_lr_on_plateau)
    warmup_steps: int = 0 # 
    warmup_ratio: float = 0.
    max_train_step: int=0
    max_eval_step: int=0
    save_checkpoint_every_step: int = 0
    update_lr_every_step: int = 0
    is_dpo: bool = False