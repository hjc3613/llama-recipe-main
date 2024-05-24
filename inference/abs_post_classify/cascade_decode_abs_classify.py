'''
@coding:utf-8
@File:DG_Web_Demo.py
@Time:2023/9/5 9:46
@Author:Papers
'''
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json
import sys
import numpy as np
import pdb

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
HUMAN_ROLE_START_TAG = "<s>human\n"
BOT_ROLE_START_TAG = "<s>bot\n"
SYSTEM_TAG = "<s>system\n你是一个医生的助手,回复病历和对话相关的内容\n"


class Cascade_Decode():
    def __init__(self, model_path):
        # init ner model
        self.model, self.tokenizer = self._init_model(model_path["abs_model"]["path"], max_length=2048,
                                                      gpu_idx=model_path["abs_model"]["gpu_idx"])
        self.gpu_idx = model_path["abs_model"]["gpu_idx"]
        self.max_token = model_path["abs_model"]["max_token"]
        pass

    def find_element(self, lst, element):
        try:
            index = lst.index(element)
            return index
        except ValueError:
            return -1

    def _init_model(self, path, max_length, gpu_idx):
        print("Load {}".format(path))
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda(gpu_idx)
        tokenizer = AutoTokenizer.from_pretrained(path,
                                                  model_max_length=max_length,
                                                  padding_side="right",
                                                  use_fast=False,
                                                  trust_remote_code=True)
        model = model.eval()
        return model, tokenizer

    def process_output(self, outputs, tokenizer, input_ids):
        response = ""
        for output in outputs.sequences:
            output = output[len(input_ids[0]):].tolist()
            # pdb.set_trace()
            output = output[:self.find_element(output, tokenizer.eos_token_id)]
            #output = output[len(input_ids):]
            response = tokenizer.decode(output)
            response = response.strip()
            return response

    def load_chat_input(self, text_file):
        # texts = []
        idxs = []
        conversations_list = []
        prefix_prompts = []
        # outputs = []
        samples = []
        with open(text_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if not (sample and type(sample) == dict):
                    print(sample)
                    continue
                conversations_list.append(sample["pred_output"])
                idxs.append(sample["id"])
               #prefix_prompts.append(sample["prefix_prompt"])
                samples.append(sample)
        return conversations_list, prefix_prompts, idxs, samples

    def load_input(self, text_file):
        texts = []
        idxs = []
        outputs = []
        samples = []
        with open(text_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if not (sample and type(sample) == dict):
                    print(sample)
                    continue
                texts.append(sample["input"])
                idxs.append(sample["id"])
                outputs.append(sample.get("output", ""))
                samples.append(sample)
        return texts, idxs, outputs, samples

    def _decode(self, model, tokenizer, prompt):
        generation_config = GenerationConfig(
            temperature=0.0,
            top_p=0.7,
            top_k=10,
            num_beams=1,
            do_sample=False,
            max_new_tokens=self.max_token,
            num_return_sequences=1,
            repetition_penalty=1.0,
            pad_token_id=151643,
            eos_token_id=151643,
            output_scores=True,
            return_dict_in_generate=True
        )
        self.model.generation_config = generation_config
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            input_ids = inputs["input_ids"].cuda(self.gpu_idx)
            outputs = model.generate(input_ids, generation_config=generation_config)
            output = self.process_output(outputs, tokenizer, input_ids)
            # 获取logit的得分
            logits = outputs.scores
            token_scores = []
            # for i in range(0, len(logits)-1):# 最后一个end不参与计算
            for i in range(0, len(logits)):# 最后一个end不参与计算
                max_prob = torch.max(torch.softmax(logits[i][0], dim=-1),dim=-1)
                token_scores.append(max_prob.values.cpu().float().numpy().item())
            average_score = np.average(token_scores)
            print("==" * 40)
            print("用户输入：", prompt)
            print("\n")
            print("模型输出：", output)
            print("模型概率：", average_score)
            return output, average_score

    def save_output(self, text_file, results, match_results, cls_resluts, result_path, texts, idxs, outputs, samples):
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(result_path, "result_" + os.path.basename(text_file))
        print(f"save result in {result_file}")
        with open(result_file, "w", encoding="utf8") as f:
            for idx, text, result, match_result, cls_reslut, output, sample in zip(idxs, texts, results, match_results,
                                                                                   cls_resluts, outputs, samples):
                sample["pred_output"] = result
                sample["match_output"] = match_result[0]
                sample["match_average_score"] = match_result[1]
                sample["cls_output"] = cls_reslut[0]
                sample["cls_average_score"] = cls_reslut[1]
                sample["gold_output"] = output
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write("\n")

    def save_output_conv(self, text_file, results, result_path, idxs, samples):
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(result_path, "result_" + os.path.basename(text_file))
        print(f"save result in {result_file}")
        with open(result_file, "w", encoding="utf8") as f:
            for idx, result, sample in zip(idxs, results, samples):
                sample["pred_output"] = result
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write("\n")

    def _process(self, input_path, result_path):
        # load data
        texts, idxs, outputs, samples = self.load_input(input_path)
        # inference
        final_results = []
        match_results = []
        cls_resluts = []
        for text in tqdm(texts, desc="inference", total=len(texts)):
            new_text =f"{SYSTEM_TAG}{HUMAN_ROLE_START_TAG}{text}{BOT_ROLE_START_TAG}"
            new_text = text
            match_result, match_average_score = self._decode(model=self.model, tokenizer=self.tokenizer, prompt=new_text)
            cls_prompt = "信息摘要：\n" + text.split("信息摘要：")[-1].strip(
                "\n") + "\n判断上面的医学信息摘要是否可以写入到门诊病历中，输出可以或者不可以。可写入到门诊病历要求：1.信息摘要的键值对无相互矛盾情况。2.信息摘要的值包含医学信息且术语化。\n"
            #cls_prompt = f"{SYSTEM_TAG}{HUMAN_ROLE_START_TAG}{cls_prompt}{BOT_ROLE_START_TAG}"
            cls_prompt = cls_prompt
            cls_reslut, cls_average_score = self._decode(model=self.model, tokenizer=self.tokenizer, prompt=cls_prompt)
            if cls_reslut == "不可以" or match_result == "否":
                final_results.append("否")
            #elif cls_average_score <= 0.95 or match_average_score <= 0.95:
            #    final_results.append("否")
            else:
                final_results.append("是")
            match_results.append((match_result,match_average_score))
            cls_resluts.append((cls_reslut,cls_average_score))
        self.save_output(input_path, final_results, match_results, cls_resluts, result_path, texts, idxs, outputs,
                         samples)

    def create_prompt(self, input_convs, tmp_pred_value_list):
        dialog_list = []
        for conv in input_convs:
            dialog_list.append(conv["value"])
        input_value = "\n".join(dialog_list)
        prompt_list = []
        for s in range(0, len(tmp_pred_value_list)):
            if tmp_pred_value_list[s].strip() == "": continue
            input_new = "判断下面的信息摘要和医患对话是否匹配或者蕴含，输出是或者否：\n医患对话：\n"
            input_new += input_value.replace(" ", "").strip()
            input_new += "\n"
            input_new += "信息摘要：\n"
            input_new += tmp_pred_value_list[s].strip()
            input_new += "\n"
            prompt_list.append((input_new, tmp_pred_value_list[s].strip()))
        return prompt_list

    def conversation_process(self, input_path, result_path):
        """
        针对离线级联效果的测试，输入为abs的输出conversation的对话格式
        """
        # load data
        conversations_list, prefix_prompts, idxs, samples = self.load_chat_input(input_path)

        # inference
        results = []

        for idx, conversations in tqdm(enumerate(conversations_list), desc="inference", total=len(conversations_list)):
            results_message = []

            for turn_idx, message in tqdm(enumerate(conversations), desc="Message", total=len(conversations)):
                start_index = max(turn_idx - 8, 0)  # 8轮次
                #if len(conversations) >= 100:
                #    continue
                convs = conversations[start_index:turn_idx + 1]
                convs = conversations[start_index:turn_idx + 5]
                #convs = conversations[start_index:turn_idx + 2]
                pred_output_list = message["pred_output"].split("\n")
                prompt_list = self.create_prompt(convs, pred_output_list)
                new_pred_output = []
                match_data = []
                cls_data = []
                #match_scores = []
                #cls_scores = []
                for prompt in prompt_list:
                    text = prompt[0]
                    abs = prompt[1]
                    if abs == "当前对话无有效医学摘要信息":
                        new_pred_output.append(abs)
                        match_average_score = 1
                        continue
                    match_result, match_average_score = self._decode(model=self.model, tokenizer=self.tokenizer, prompt=text)
                    cls_prompt = "信息摘要：\n" + text.split("信息摘要：")[-1].strip(
                        "\n") + "\n判断上面的医学信息摘要是否可以写入到门诊病历中，输出可以或者不可以。可写入到门诊病历要求：1.信息摘要的键值对无相互矛盾情况。2.信息摘要的值包含医学信息且术语化。\n"
                    cls_reslut, cls_average_score = self._decode(model=self.model, tokenizer=self.tokenizer, prompt=cls_prompt)
                    #if cls_reslut == "不可以" or match_result == "否": continue
                    #new_pred_output.append(prompt[1])
                    match_data.append({"prompt":text,"abs":abs,"match_reslut":match_result, "match_score": match_average_score})
                    cls_data.append({"prompt":cls_prompt,"abs":abs,"cls_reslut":cls_reslut, "cls_score": cls_average_score})
                    if cls_reslut == "不可以" or match_result == "否": continue
                    #if cls_average_score <=0.95 or match_average_score <= 0.95: continue
                    new_pred_output.append(prompt[1])
                message["match_data"] = match_data
                message["cls_data"] = cls_data
                message["filter_pred_output"] = "\n".join(new_pred_output)
                message["filter_pred_output_scores"] = match_average_score
                results_message.append(message)
            results.append(results_message)
            if idx==10:
                self.save_output_conv(input_path, results, result_path, idxs, samples)
        self.save_output_conv(input_path, results, result_path, idxs, samples)


if __name__ == '__main__':
    infile = sys.argv[1]
    gpu_num = sys.argv[2]
    result_path = sys.argv[3]
    hf_home = sys.argv[4]
    max_token = int(sys.argv[5])
    mode = sys.argv[6]
    model_path = {
        "abs_model": {
            "path": hf_home,
            "gpu_idx": int(gpu_num),
            "max_token": max_token},
    }
    web_demo = Cascade_Decode(model_path)
    if mode == "chat":
        web_demo.conversation_process(infile, result_path)
    else:
        web_demo._process(infile, result_path)
