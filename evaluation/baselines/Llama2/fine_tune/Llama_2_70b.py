import torch
from peft import PeftModel, PeftConfig
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

from evaluation.baselines.prompt import get_prompt


class PELlama2Lora70B(object):
    def __init__(self, cuda_ip=None):
        self.model_path = '/path/to/initial/Llama-2-70b-chat-hf'  # initial model
        self.lora_path = '/path/to/fine-tuned/Llama-2-70b-lora'  # LORA model

        if cuda_ip:
            self.cuda_list = str(cuda_ip).split(',')
        else:
            self.cuda_list = '0,1,2,3'.split(',')

        self.memory = '47GiB'
        self.max_memory = {int(cuda): self.memory for cuda in self.cuda_list}

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        # set pad token id
        self.tokenizer.pad_token = self.tokenizer.eos_token_id

        # load base model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=self.max_memory)
        print('base model loaded')

        # load lora
        device_map = {f"base_model.model.{k}": v for k, v in self.model.hf_device_map.items()}
        self.lora_model = PeftModel.from_pretrained(
            self.model,
            self.lora_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16)
        self.lora_model.eval()
        print('lora model loaded')

    def generate(self, input_text, max_length=512):
        assert isinstance(input_text, str)
        input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'].to(self.model.device)

        output_ids = self.lora_model.generate(**{'input_ids': input_ids, 'max_new_tokens': max_length})

        output = self.tokenizer.decode(output_ids[:, input_ids.shape[1]:][0], skip_special_tokens=True)

        return output

    def batch_generate(self, input_text_list, max_length=512):  # 2048
        assert isinstance(input_text_list, list)

        input_ids = self.tokenizer.batch_encode_plus(input_text_list, return_tensors='pt', padding="longest", truncation=True).to(self.model.device)

        output_ids = self.lora_model.generate(**{'input_ids': input_ids['input_ids'], 'attention_mask': input_ids['attention_mask'],
                                              'max_new_tokens': max_length, 'do_sample': False})

        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)

        return outputs

    def predict(self, paragraph: str):  # 必须是这个格式
        prompt = get_prompt()
        input_text = prompt.format(paragraph)
        return self.generate(input_text)

    def batch_predict(self, paragraphs: list):
        prompt = get_prompt()
        input_texts = [prompt.format(paragraph) for paragraph in paragraphs]
        return self.batch_generate(input_texts)
