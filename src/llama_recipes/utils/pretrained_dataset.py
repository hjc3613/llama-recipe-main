import time
import torch
import logging
from .supervised_dataset import SupervisedDataset
from .indexed_dataset import MMapIndexedDatasetBuilder, MMapIndexedDataset
from transformers import AutoTokenizer
import numpy as np
import glob
import re
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def create_bin_ds(args, ):
    """save idx and bin to disk"""
    train_path, tokenizer_path, seq_length = args.input, args.tokenizer_path, args.seq_length
    startup_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenized_dataset = SupervisedDataset(train_path, tokenizer=tokenizer, padding='longest', max_length=seq_length, num_process=args.num_process)
    
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    level = "document"
    # if self.args.split_sentences:
    #     level = "sentence"

    logger.info("Vocab size: %s", tokenizer.vocab_size)
    logger.info("Output prefix: %s", args.output_prefix)
    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}_{level}.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}_{level}.idx"
        # vocab_size=None : use int32 dtype for -100 will be used in labels
        builders[key] = MMapIndexedDatasetBuilder(output_bin_files[key])
    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    logger.info("Time to startup:%s", startup_end - startup_start)

    skip_num = 0
    for i, doc in enumerate(iter(tokenized_dataset), start=1):
        for key in args.json_keys:
            sentences = [doc[key]]
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                if args.seq_length is not None and len(sentence) > args.seq_length:
                    skip_num += 1
                    continue

                total_bytes_processed += len(sentence) * torch.int32.itemsize
                builders[key].add_item(sentence)
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            logger.info("Processed %s documents (%s docs/s, %s MB/s).", i, i / elapsed, mbs)

    logger.info("Skip %s sample exceeded seq-length(%s)", skip_num, args.seq_length)
    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


class IsNotValidError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "Expression is not valid"
def ensure_valid(expression, error_message=None):
    if not expression:
        raise IsNotValidError(error_message)

class MTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        documents,
    ):
        # Params to store.
        self.name = name

        # Dataset.
        self.packed_indexed_dataset = get_packed_indexed_dataset(data_prefix)

        # Checks
        ensure_valid(np.min(documents) >= 0)
        ensure_valid(len(self.packed_indexed_dataset) > 0)

        self.length = len(list(self.packed_indexed_dataset.values())[0])

        ensure_valid(np.max(documents) < self.length)
        for dataset in self.packed_indexed_dataset.values():
            if len(dataset) != self.length:
                raise Exception("Dimension is not correct !")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        packed_data = dict()
        for key, dataset in self.packed_indexed_dataset.items():
            packed_data[key] = dataset.get(idx)
            ensure_valid(len(packed_data[key]) > 0)
        return packed_data

def get_packed_indexed_dataset(data_prefix: str):
    index_dataset_name = f"{data_prefix}_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()
    for field in all_field:
        packed_dataset[field] = MMapIndexedDataset(f"{data_prefix}_{field}_document")
    return packed_dataset

def build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    # train_valid_test_num_samples,
    # seed,
):
    """Build train, valid, and test datasets."""

    
    # Only Support Single dataset.
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
        data_prefix=data_prefix,
        splits_string=splits_string,
        seq_length=seq_length,
        # train_valid_test_num_samples=train_valid_test_num_samples,
        # seed=seed,
    )

    return all_train_datasets, all_valid_datasets, all_test_datasets

def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index

def _build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    # train_valid_test_num_samples,
    # seed,
):
    """Build train, valid, and test datasets."""

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)

    total_num_of_documents = len(list(packed_indexed_dataset.values())[0])
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                documents=documents,
                seq_length=seq_length,
                # num_samples=train_valid_test_num_samples[index],
                # seed=seed
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class DecoderPackedMTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        documents,
        # num_samples,
        seq_length: int,
        # seed,
    ):
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, documents=documents)

        self.seq_length = seq_length


    def __len__(self):
        return len(self.mtf_dataset)

    def __getitem__(self, idx):
        item = self.mtf_dataset[idx]
        return {
            "input_ids": self._cut_token(item["input_ids"], np.int64),
            # "attention_mask": self._cut_token(item["attention_mask"], np.int64),
            "labels": self._cut_token(item["labels"], np.int64),
        }
    
    def _cut_token(self, token, dtype):
        token_length = len(token)
        if token_length >= self.seq_length:
            token = token[:self.seq_length]
        return token.astype(dtype)
    

