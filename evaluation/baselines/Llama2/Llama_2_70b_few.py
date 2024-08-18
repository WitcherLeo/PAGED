import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from evaluation.baselines.prompt import get_prompt


class PELlama2For70BFew(object):  # few shot
    def __init__(self, cuda_ip=None):
        self.model_path = '/path/to/Llama-2-70b-chat-hf'
        if cuda_ip:
            self.cuda_list = str(cuda_ip).split(',')
        else:
            self.cuda_list = '0,1,2,3'.split(',')
        self.memory = '40GiB'
        self.max_memory = {int(cuda): self.memory for cuda in self.cuda_list}

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        # set pad token id
        self.tokenizer.pad_token = self.tokenizer.eos_token_id

        # load base model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=self.max_memory)
        self.model.eval()
        print('model loaded')

    def generate(self, input_text, max_length=2048):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.model.device)

        output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=False)  # Content Length: 4k

        output = self.tokenizer.decode(output_ids[:, input_ids.shape[1]:][0], skip_special_tokens=True)

        return output

    def batch_generate(self, input_texts, max_length=2048):
        input_ids = self.tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding="longest", truncation=True).to(self.model.device)

        output_ids = self.model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                         max_length=max_length, do_sample=False)

        output = self.tokenizer.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)

        return output

    def predict(self, paragraph: str):  # 必须是这个格式
        prompt = get_prompt()
        input_text = prompt.format(paragraph)
        return self.generate(input_text)

    def batch_predict(self, paragraphs: list):
        prompt = get_prompt()
        input_texts = [prompt.format(paragraph) for paragraph in paragraphs]
        return self.batch_generate(input_texts)

