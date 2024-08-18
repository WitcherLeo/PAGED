import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, T5Tokenizer, AutoModelForSeq2SeqLM

from evaluation.baselines.prompt import get_prompt


class PEFlanT5LoraXXL(object):
    def __init__(self, cuda_ip=None):
        self.model_path = '/path/to/initial/flan-t5-xxl'  # initial model path
        self.lora_path = '/path/to/saved/fine-tuned/flan-t5-xxl'  # LORA model path

        if cuda_ip:
            self.cuda_list = str(cuda_ip).split(',')
        else:
            self.cuda_list = '0'.split(',')
        self.memory = '45GiB'
        self.max_memory = {int(cuda): self.memory for cuda in self.cuda_list}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # set pad token id
        self.tokenizer.pad_token = self.tokenizer.eos_token_id

        # load base model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
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

    def generate(self, text, max_length=1024):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

        outputs = self.lora_model.generate(**{'input_ids': input_ids, 'max_length': max_length, 'num_beams': 4})

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, texts, max_length=1024):
        input_ids = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest", truncation=True).to(self.model.device)

        outputs = self.lora_model.generate(**{'input_ids': input_ids['input_ids'], 'attention_mask': input_ids['attention_mask'],
                                              'max_length': max_length, 'num_beams': 4})

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def batch_predict(self, paragraphs: list):
        prompt = get_prompt()
        input_texts = [prompt.format(paragraph) for paragraph in paragraphs]
        return self.batch_generate(input_texts)

