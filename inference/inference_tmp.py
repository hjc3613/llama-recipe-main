import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, Qwen2ForCausalLM
# from src.llama_recipes.qwen.modeling_qwen import QWenLMHeadModel

model_id = "/fl-ift/med/common/Qwen1.5-14B-Chat/"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=False,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            # use_flash_attn=True,
)
generation_config = GenerationConfig.from_pretrained(model_id,trust_remote_code=True)

def generate(text):
    messages = [
    # {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print('注意当前chat_format=','llama', '请检查是否与训练格式保持一致')
    encoded_input = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    encoded_input = encoded_input.to(model.device)
    generated_ids = model.generate(
    **encoded_input,
    max_new_tokens=512,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.decode(generated_ids[0][len(encoded_input['input_ids'][0]):]).replace('</s>', '').replace('<s>', '')
    decoded_output = decoded_output.replace('<|endoftext|>', '').replace('<|im_end|>', '')
    return decoded_output

res = generate("hello, who are you?")
print(res)