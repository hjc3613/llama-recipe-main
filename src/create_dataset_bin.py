from llama_recipes.utils.pretrained_dataset import create_bin_ds,build_train_valid_test_datasets
from torch.utils.data.dataloader import DataLoader
import os

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input JSONL', default='/fl-ift/med/common/datasets/med/preprocessed2')
    parser.add_argument('--json-keys', nargs='+', default=['input_ids', 'labels'], 
                        help='space separate listed of keys to extract from json')
    parser.add_argument("--tokenizer_path", type=str, default='/fl-ift/med/common/Qwen-14B-Base',
                       help="Name or path of the huggingface tokenizer.")
    parser.add_argument('--seq-length', type=int, default=4096,
                       help='Maximum sequence length to process.')
    parser.add_argument('--output-prefix', type=str, default='tmp',
                       help='Path to binary output file without suffix')
    parser.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    parser.add_argument('--num_process', type=int, default=1, 
                        help='process nums to tokenize the input file')
    args = parser.parse_args()
    args.output_prefix = args.input.split('.')[0]+'_bin/dataset_packed'
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    create_bin_ds(args=args)
    # train_ds, _, _ = build_train_valid_test_datasets(args.output_prefix, splits_string='100,0,0', seq_length=4096)
    ...

    