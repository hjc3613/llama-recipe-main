'''
@coding:utf-8
@File:supervised_dataset_base.py
@Time:2023/11/4 11:10
@Author:Papers
'''
import torch
from torch.utils.data import Dataset
import logging
import transformers
from typing import Dict, Sequence
import copy
import json
import io
import pandas as pd
IGNORE_INDEX = -100

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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, index=False, seq_len=False):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_file(data_path)

        logging.warning("Formatting inputs...")
        sources = [
            example['input']
            for example in list_data_dict
        ]

        if 'output' in list_data_dict[0]:
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        elif 'target' in list_data_dict[0]:
            targets = [f"{example['target']}{tokenizer.eos_token}" for example in list_data_dict]
        elif 'label' in list_data_dict[0]:
            targets = [f"{example['label']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.index = index
        self.seq_len = seq_len
        if self.index:
            self.indexes = torch.tensor([i['index'] for i in list_data_dict])
        if self.seq_len:
            self.seq_len_list = torch.tensor([len(i) for i in self.input_ids])

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self,
                   sources: Sequence[str],
                   targets: Sequence[str],
                   tokenizer: transformers.PreTrainedTokenizer,
                   ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in
                                                 (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if self.index:
        #     return dict(input_ids=self.input_ids[i], labels=self.labels[i], index=self.indexes[i])
        # else:
        #     return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        example = dict(
            input_ids=self.input_ids[i], 
            labels=self.labels[i],
            index=self.indexes[i] if self.index else None,
            seq_len = self.seq_len_list[i] if self.seq_len else None
        )
        example = {k:v for k,v in example.items() if v is not None}
        return example
