import os

def validate_train_args(train_args):
    assert isinstance(train_args.context_length, int) and train_args.context_length > 5
    assert os.path.exists(train_args.model_name) and os.path.isdir(train_args.model_name) 
    assert train_args.batching_strategy == 'padding'
    assert isinstance(train_args.gradient_clipping, bool) and train_args.gradient_clipping == True
    