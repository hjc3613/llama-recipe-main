
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import logging
import transformers
import json
import io
import pandas as pd
from llama_recipes.qwen_dpo.dpo_utils import concatenated_inputs

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return concatenated_inputs(padded_batch)
    return collate_fn


def tokenize_batch_element(
        prompt: str, 
        chosen: str, 
        rejected: str, 
        # truncation_mode: str, 
        tokenizer:transformers.AutoTokenizer, 
        # max_length: int, 
        # max_prompt_length: int
    ) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    # longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    # if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
    #     if truncation_mode == 'keep_start':
    #         prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
    #     elif truncation_mode == 'keep_end':
    #         prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
    #     else:
    #         raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    # if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
    #     chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
    #     rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    # 键值对顺序很重要，在collate_fn中会按chosen、rejected顺序进行拼接，然后在model.forward()中根据先后顺序再次分离chosen和rejected
    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload_v2(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = []
    for line in f:
        jdict.append(json.loads(line))
    f.close
    return jdict

def xlsload(f):
    df = pd.read_excel(f)
    result = [dict(row) for idx, row in df.iterrows()]
    return result

def load_file(file:str):
    if file.endswith('.jsonl'):
        return jload_v2(file)
    elif file.endswith('.xlsx'):
        return xlsload(file)


class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            prompt_col='input', 
            chosen_col='chosen', 
            reject_col='reject',
            # ref_chosen_score_col='ref_chosen_score',
            # ref_reject_score_col='ref_reject_score',
            # index_col='index',
            # mode='train'
        ):
        super(DPODataset, self).__init__()
        self.examples = []
        logging.info("Loading data...")
        list_data_dict = load_file(data_path)
        assert prompt_col in list_data_dict[0], f'dataset缺少{prompt_col}列'
        assert chosen_col in list_data_dict[0], f'dataset缺少{chosen_col}列'
        assert reject_col in list_data_dict[0], f'dataset缺少{reject_col}列'
        # if mode == 'train':
            # assert index_col in list_data_dict[0], f'dataset缺少{index_col}列'
            # assert ref_reject_score_col in list_data_dict[0], f'dataset缺少{ref_reject_score_col}列'
        for idx, example in enumerate(list_data_dict):
            batch = tokenize_batch_element(prompt=example[prompt_col], chosen=example[chosen_col], rejected=example[reject_col], tokenizer=tokenizer)
            # batch[ref_chosen_score_col] = example.get(ref_chosen_score_col, 0)
            # batch[ref_reject_score_col] = example.get(ref_reject_score_col, 0)
            batch['index'] = idx
            self.examples.append(batch)
        ...
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.examples[i]
