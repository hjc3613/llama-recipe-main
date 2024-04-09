from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn

def pruning_qw(path, selected_layers):
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path,trust_remote_code=True)
    model.transformer.h = nn.ModuleList([block for i, block in enumerate(model.transformer.h) if i in selected_layers])
    config.num_hidden_layers = len(selected_layers)
    model.save_pretrained(path+'_pruning_middle_ou')
    tokenizer.save_pretrained(path+'_pruning_middle_ou')
    config.save_pretrained(path+'_pruning_middle_ou')

if __name__ == '__main__':
    path = '/fl-ift/med/common/quant_test_model_0319_1250_abstract/hf'
    selected_layers = [
0,
1,
2,
3,
4,
5,
6,
# 7,
8,
# 9,
10,
# 11,
12,
# 13,
14,
# 15,
16,
# 17,
18,
# 19,
20,
# 21,
22,
# 23,
24,
# 25,
26,
# 27,
28,
# 29,
30,
# 31,
32,
# 33,
34,
# 35,
36,
# 37,
38,
# 39,
40,
# 41,
42,
# 43,
44,
# 45,
46,
# 47,
48,
# 49,
50,
# 51,
52,
# 53,
54,
# 55,
56,
# 57,
58,
# 59,
60,
# 61,
62,
# 63,
64,
# 65,
66,
# 67,
68,
# 69,
70,
# 71,
72,
# 73,
74,
# 75,
76,
77,
78,
79,
    ]

    pruning_qw(path=path, selected_layers=selected_layers)