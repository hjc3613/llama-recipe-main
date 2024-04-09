
from transformers.modeling_utils import shard_checkpoint, _add_variant
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME, 
    SAFE_WEIGHTS_NAME, 
    WEIGHTS_INDEX_NAME, 
    WEIGHTS_NAME,
)
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
import os
import re
from transformers.utils import logging
import json

logger = logging.get_logger(__name__)

def save_state_dict_shard(state_dict, save_directory, max_shard_size='5GB', safe_serialization=True):
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in shards.keys()
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    for shard_file, shard in shards.items():
        if safe_serialization:
            # At some point we will need to deal better with save_function (used for TPU and other distributed
            # joyfulness), but for now this enough.
            safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
        else:
            raise Exception('只支持safetensors保存checkpoints')

    if index is None:
        path_to_weights = os.path.join(save_directory, weights_name)
        logger.info(f"Model weights saved in {path_to_weights}")
    else:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        # Save the index as well
        with open(os.path.join(save_directory, save_index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )