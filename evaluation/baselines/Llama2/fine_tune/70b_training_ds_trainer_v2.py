import os
import random
import sys
import numpy as np
import pynvml
import torch
import logging
import datasets

from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, EvalPrediction
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from huggingface_hub import HfFolder
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

import argparse
import bitsandbytes as bnb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tools.nltk_tools.nltk_tool import NLTKTool
from evaluation.baselines.prompt import get_prompt_for_train


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )


def get_gpu_mem_info(gpu_id=0):
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 0)
    used = round(meminfo.used / 1024 / 1024, 0)
    free = round(meminfo.free / 1024 / 1024, 0)

    logging.info('-----------------------gpu memory info----------------------------')
    logging.info('gpu_id: %d, total: %dMiB, used: %dMiB, free: %dMiB' % (gpu_id, total, used, free))

    return total, used, free


def batch_sentences_eval(decoded_labels, decoded_preds):
    pass


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    # parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    # parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    # parser.add_argument(
    #     "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    # )
    # # add training hyperparameters for epochs, batch size, learning rate, and seed
    # parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    # parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    # parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    # parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing.")
    parser.add_argument("--bf16", type=bool, default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    # parser.add_argument("--hf_token", type=str, default=HfFolder.get_token(), help="Token to use for uploading models to Hugging Face Hub.")
    args = parser.parse_known_args()
    return args


def train(train_dataset, dev_dataset, test_dataset, read_model_path, model_save_path, save_model):
    """
    fine-tune the Llama2 model in supervised learning mode
    """
    # seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # dataset preprocessing
    def preprocess_function(samples):
        paragraphs = samples['paragraph']  # list of strings
        procedure_tuples_texts = samples['procedure_tuples_text']  # list of strings

        assert len(paragraphs) == len(procedure_tuples_texts)

        # add prompt and construct model training text
        prompt = get_prompt_for_train()
        texts = [prompt.format(paragraph, procedure_tuples_texts[index]) for index, paragraph in enumerate(paragraphs)]
        return texts

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # cuda settings
    cuda_list = '0,1,2,3'.split(',')
    memory = '47GiB'
    max_memory = {int(cuda): memory for cuda in cuda_list}

    # model
    model = LlamaForCausalLM.from_pretrained(read_model_path,
                                             load_in_8bit=True,
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             max_memory=max_memory)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = LlamaTokenizer.from_pretrained(read_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    logging.info('\n\n')
    logging.info('---------------------------model loaded-------------------------------\n')

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=128,
        lora_alpha=16,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    # print_trainable_parameters(model)

    # Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
    training_args = TrainingArguments(
        output_dir=model_save_path,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,  # use gradient_checkpointing to 减少内存但拖慢训练
        max_grad_norm=0.3,
        num_train_epochs=10,
        learning_rate=2e-4,
        bf16=True,

        # logging & evaluation strategies
        logging_dir=f"logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=99,
        load_best_model_at_end=False,  # load_best_model_at_end requires the save and eval strategy to match
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    nltk_tool = NLTKTool()

    # metric used for LM prediction evaluation
    def compute_metrics(p: EvalPrediction):
        preds, labels = p

        # decode predictions and labels
        # predictions = np.argmax(predictions, axis=2)  # why need this step, because the output is logits
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, device="cpu")
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, device="cpu")

        # compute average BLEU score
        total_bleu = 0
        for i in range(len(decoded_labels)):
            total_bleu += nltk_tool.compute_bleu_single(decoded_preds[i], decoded_labels[i])

        avg_bleu = total_bleu / len(decoded_labels)
        avg_bleu = round(avg_bleu, 4)

        result = {"BLEU": avg_bleu}

        logging.info('\n\n')
        logging.info('---------------------------compute_metrics-------------------------------')
        logging.info(result)
        logging.info('\n\n')

        return result

    # args, _ = parse_arge()

    # Define training args
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        max_seq_length=2048,
        formatting_func=preprocess_function,
        args=training_args
    )

    logging.info('\n\n')
    logging.info('------------------------************------------------------')
    logging.info('------------------------trainer loaded, start training... ---------------')
    logging.info('------------------------************------------------------')

    # Start training
    trainer.train()

    logging.info('\n\n')
    logging.info('------------------------************------------------------')
    logging.info('------------------------training finished---------------------------')
    logging.info('------------------------************------------------------')

    # save model
    if save_model:
        # Save our tokenizer and create model card
        tokenizer.save_pretrained(model_save_path)
        trainer.model.save_pretrained(model_save_path)
        # trainer.save_model(model_save_path)
        logging.info('\n\n')
        logging.info('---------------------------Save tokenizer and final model----------------------------')

    # test
    predictions = trainer.predict(test_dataset=test_dataset)
    # print(predictions.predictions.shape, predictions.label_ids.shape)

    test_result = compute_metrics(predictions)
    logging.info('\n\n')
    logging.info('---------------------------test result-------------------------------')
    logging.info(test_result)
    logging.info('---------------------------------------------------------------------')
    logging.info('\n\n')


def train_main(save_model):
    # logging
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    read_model_path = '/path/to/initial/llama2-70b-chat-hf'

    model_save_path = '/path/to/save/fine_tuned/llama2-70b-chat-hf'

    # read dataset
    read_dataset_path = 'evaluation/baselines/training_data/'
    train_data_path = read_dataset_path + 'train'
    dev_data_path = read_dataset_path + 'dev'
    test_data_path = read_dataset_path + 'test'

    train_dataset = datasets.load_from_disk(train_data_path)
    dev_dataset = datasets.load_from_disk(dev_data_path)
    test_dataset = datasets.load_from_disk(test_data_path)

    logging.info('train data size: {}'.format(len(train_dataset)))
    logging.info('dev data size: {}'.format(len(dev_dataset)))
    logging.info('test data size: {}'.format(len(test_dataset)))

    train(train_dataset, dev_dataset, test_dataset, read_model_path, model_save_path, save_model)


if __name__ == '__main__':
    train_main(save_model=True)
