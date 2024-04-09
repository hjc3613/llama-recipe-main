import os
from enum import Enum, unique, auto
# model = 'qw'
FT_MODEL_TYPE='FT_MODEL_TYPE'
model = os.getenv(FT_MODEL_TYPE)
model = str(model).lower()
@unique
class LlamaType(Enum):
    qw = auto()
    qwdpo = auto()
    mixtral = auto()
    qw2 = auto()
    llama = auto()
    qwdistillation = auto()

if model==LlamaType.qw.name:
    from llama_recipes.qwen.modeling_qwen import QWenBlock as LlamaDecoderLayer
    from llama_recipes.qwen.modeling_qwen import QWenLMHeadModel as LlamaForCausalLM
    from llama_recipes.qwen.tokenization_qwen import QWenTokenizer as LlamaTokenizer
    from llama_recipes.qwen.configuration_qwen import QWenConfig as LlamaConfig
elif model==LlamaType.qwdistillation.name:
    from llama_recipes.qwen_distillation.modeling_qwen_distillation import QWenBlock as LlamaDecoderLayer
    from llama_recipes.qwen_distillation.modeling_qwen_distillation import QWenLMHeadModelDistillation as LlamaForCausalLM
    from llama_recipes.qwen_distillation.tokenization_qwen import QWenTokenizer as LlamaTokenizer
    from llama_recipes.qwen_distillation.configuration_qwen import QWenConfig as LlamaConfig
elif model==LlamaType.mixtral.name:
    from transformers import MixtralConfig as LlamaConfig
    from transformers import MixtralForCausalLM as LlamaForCausalLM
    from transformers import LlamaTokenizer
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer as LlamaDecoderLayer
elif model==LlamaType.qw2.name:
    from transformers import Qwen2ForCausalLM as LlamaForCausalLM
    from transformers import Qwen2Config as LlamaConfig
    from transformers import Qwen2Tokenizer as LlamaTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as LlamaDecoderLayer
elif model==LlamaType.qwdpo.name:
    from llama_recipes.qwen_dpo.modeling_qwen_dpo import QWenBlock as LlamaDecoderLayer
    from llama_recipes.qwen_dpo.modeling_qwen_dpo import QWenLMHeadModelDPO as LlamaForCausalLM
    from llama_recipes.qwen_dpo.tokenization_qwen import QWenTokenizer as LlamaTokenizer
    from llama_recipes.qwen_dpo.configuration_qwen import QWenConfig as LlamaConfig
elif model == LlamaType.llama.name:
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
else:
    raise Exception(f'请设置环境变量: {FT_MODEL_TYPE},其值必须属于{[i.name for i in LlamaType]},当前{FT_MODEL_TYPE}={model}')