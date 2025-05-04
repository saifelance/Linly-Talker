import os
import torch
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import ip, api_port, model_path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Linly:
    def __init__(self, mode='api', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf", prefix_prompt='''请用少于25个字回答以下问题\n\n'''):
        self.url = f"http://{ip}:{api_port}"  # FastAPI endpoint for API mode
        self.headers = { "Content-Type": "application/json" }
        self.data = { "question": "北京有什么好玩的地方？" }

        self.prefix_prompt = prefix_prompt
        self.mode = mode
        self.history = []

        self.is_t5 = False  # Will be auto-set based on model type

        if mode != 'api':
            self.model, self.tokenizer = self.init_model(model_path)

    def init_model(self, path="Linly-AI/Chinese-LLaMA-2-7B-hf"):
        if "t5" in path.lower() or "flan" in path.lower():
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(path)
            tokenizer = T5Tokenizer.from_pretrained(path)
            self.is_t5 = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
            self.is_t5 = False
        return model, tokenizer

    def message_to_prompt(self, message, system_prompt=""):
        system_prompt = self.prefix_prompt + system_prompt
        for user, bot in self.history:
            system_prompt += f" User: {user.strip()} Bot: {bot.strip()}"
        prompt = f"{system_prompt} ### Instruction:{message.strip()}  ### Response:"
        return prompt

    def generate(self, question, system_prompt=""):
        if self.mode != 'api':
            prompt = self.message_to_prompt(question, system_prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if not self.is_t5:
                inputs = inputs.to("cuda:0" if torch.cuda.is_available() else "cpu")
            try:
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=512 if self.is_t5 else 2048,
                    do_sample=not self.is_t5,
                    top_k=20 if not self.is_t5 else None,
                    top_p=0.84 if not self.is_t5 else None,
                    temperature=1 if not self.is_t5 else None,
                    repetition_penalty=1.15 if not self.is_t5 else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
                if not self.is_t5:
                    response = response.split("### Response:")[-1]
                return response.strip()
            except Exception as e:
                print("Error during generation:", e)
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict_api(question)

    def predict_api(self, question):
        self.data["question"] = question
        try:
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps({ "prompt": question }))
            return response.json().get('response', '[API Error: No response]')
        except Exception as e:
            print("API error:", e)
            return "[API Error]"

    def chat(self, system_prompt, message, history):
        self.history = history
        response = self.generate(message, system_prompt)
        self.history.append([message, response])
        return response, self.history

    def clear_history(self):
        self.history = []

def test():
    llm = Linly(mode='offline', model_path='MBZUAI/LaMini-Flan-T5-783M')
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
