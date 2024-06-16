# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from transformers.optimization import get_cosine_schedule_with_warmup, get_scheduler
from transformers import (
    # LlamaForCausalLM,
    # LlamaTokenizer,
    # LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# from qwen.modeling_qwen import QWenBlock as LlamaDecoderLayer
# from qwen.modeling_qwen import QWenLMHeadModel as LlamaForCausalLM
# from qwen.tokenization_qwen import QWenTokenizer as LlamaTokenizer
# from qwen.configuration_qwen import QWenConfig as LlamaConfig

from llama_recipes.import_llama import (
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaTokenizer,
)

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.pretrained_dataset import build_train_valid_test_datasets
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers_for_qwen,
    freeze_transformer_layers,
    active_transformer_layers_for_qwen,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

from llama_recipes.utils.supervised_dataset import SupervisedDataset
from llama_recipes.utils.dpo_dataset import DPODataset, get_collate_fn as dpo_collate_fn
from llama_recipes.utils.validate_parameters import validate_train_args

def create_scheduler(train_config, optimizer):
    get_scheduler(name=train_config.scheduler)

def get_train_val_dataset(train_config:TRAIN_CONFIG, tokenizer):
    if train_config.dataset_format=='bin':
        dataset_train, _, _ = build_train_valid_test_datasets(
            train_config.dataset, 
            splits_string='100,0,0', 
            seq_length=train_config.context_length
        )
        if train_config.run_validation and train_config.val_ds:
            dataset_val, _, _ = build_train_valid_test_datasets(
                train_config.val_ds, 
                splits_string='100,0,0', 
                seq_length=train_config.context_length
            )
        else:
            dataset_val = None
    else:
        if train_config.is_dpo:
            dataset_train = DPODataset(train_config.dataset, tokenizer)
            if train_config.run_validation and train_config.val_ds:
                dataset_val = DPODataset(train_config.val_ds, tokenizer)
            else:
                dataset_val = None
        else:
            dataset_train = SupervisedDataset(train_config.dataset, tokenizer, max_length=train_config.context_length)
            if train_config.run_validation and train_config.val_ds:
                dataset_val = SupervisedDataset(train_config.val_ds, tokenizer, max_length=train_config.context_length)
            else:
                dataset_val = None
    return dataset_train, dataset_val

def main(**kwargs):

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    validate_train_args(train_config)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print('\nlocal_rank: ', local_rank, 'rank: ', rank, 'word_size: ', world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        model_name_lower = train_config.model_name.lower()
        if 'qwen' in model_name_lower and 'qwen1.5' not in model_name_lower and 'qwen2' not in model_name_lower:
            flash_attn_args = {
                'use_flash_attn':True
            }
        else:
            flash_attn_args = {
                'attn_implementation':'flash_attention_2'
            }
        llama_config = LlamaConfig.from_pretrained(
                train_config.model_name, 
                # attn_implementation="flash_attention_2",
                # use_flash_attn=True,
                torch_dtype=torch.bfloat16,
                **flash_attn_args,
                )
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                # attn_implementation="flash_attention_2",
                # use_flash_attn=True,
                torch_dtype=torch.bfloat16,
                **flash_attn_args,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(
                train_config.model_name, 
                # attn_implementation="flash_attention_2",
                # use_flash_attn=True,
                torch_dtype=torch.bfloat16,
                **flash_attn_args,
                )
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)
            

    else:
        from accelerate import infer_auto_device_map, init_empty_weights,load_checkpoint_and_dispatch
        llama_config = LlamaConfig.from_pretrained(train_config.model_name)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(llama_config)
            # device_map = infer_auto_device_map(model, no_split_module_classes=model._no_split_modules, max_memory=['10GB']*4)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=train_config.model_name, device_map="auto", no_split_module_classes=model._no_split_modules
        )       
        # model = LlamaForCausalLM.from_pretrained(
        #     train_config.model_name,
        #     load_in_8bit=True if train_config.quantization else None,
        #     device_map="auto" if train_config.quantization else 'auto',
        #     use_cache=use_cache,
        # )
    # model.model.gradient_checkpointing = True
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            # from optimum.bettertransformer import BetterTransformer
            # model = BetterTransformer.transform(model)
            pass
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            if train_config.freeze_reverse:
                active_transformer_layers_for_qwen(model, train_config.freeze_strategy)
            else:
                freeze_transformer_layers(model, train_config.num_freeze_layers)
                # freeze_transformer_layers_for_qwen(model, train_config.num_freeze_layers, train_config.freeze_strategy)
            

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
        print('fsdp model: ', model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        # model.to("cuda")
        ...
    '''
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)
    '''
    # 加载数据
    # dataset_train = SupervisedDataset(train_config.dataset, tokenizer, max_length=train_config.context_length)
    # if train_config.run_validation and train_config.val_ds:
    #     dataset_val = SupervisedDataset(train_config.val_ds, tokenizer, max_length=train_config.context_length)
    dataset_train, dataset_val = get_train_val_dataset(train_config, tokenizer)
        
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    if train_config.is_dpo:
        train_dl_kwargs['collate_fn'] = dpo_collate_fn(tokenizer=tokenizer)

    # Create DataLoaders for the training and validation dataset
    train_dataloader = DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation and train_config.val_ds:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
        if train_config.is_dpo:
            val_dl_kwargs['collate_fn'] = dpo_collate_fn(tokenizer=tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
    
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    elif fsdp_config.optimizer == 'PagedAdamW32bit':
        from bitsandbytes.optim import PagedAdamW32bit, AdamW8bit
        optimizer = AdamW8bit(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        # optimizer = optim.AdamW(
        #     [
        #         {'params':[p for name, p in model.named_parameters() if 'wte' in name], 'lr':train_config.lr / 100},
        #         {'params':[p for name, p in model.named_parameters() if 'wte' not in name], 'lr':train_config.lr},
        #     ],
        #     weight_decay=train_config.weight_decay,
        # )
    print('optimizer type: ', type(optimizer))
    if train_config.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    else:
        total_steps = train_config.num_epochs * len(train_dataloader)//train_config.gradient_accumulation_steps
        if train_config.max_train_step>0:
            total_steps = min(total_steps, train_config.max_train_step)
        
        print('total_steps: ', total_steps, 'len(dataset_train): ', len(dataset_train))
        if train_config.warmup_steps > 0 and train_config.warmup_ratio > 0:
            warmup_steps = min(train_config.warmup_steps, int(train_config.warmup_ratio*total_steps))
            print(f'warmup_steps: min({train_config.warmup_steps} and {train_config.warmup_ratio*total_steps})=',warmup_steps)
        elif train_config.warmup_ratio > 0:
            warmup_steps = int(train_config.warmup_ratio*total_steps)
            print(f'warmup_steps {train_config.warmup_ratio}*{total_steps}=', warmup_steps)
        elif train_config.warmup_steps > 0:
            warmup_steps = train_config.warmup_steps
        else:
            warmup_steps=0
        print('warmup_steps: ', warmup_steps)
        # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
        scheduler = get_scheduler(name=train_config.scheduler, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('scheduler type: ', type(scheduler))

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
