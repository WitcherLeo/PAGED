import os
import random
import sys

import numpy as np
import pynvml
import torch
import logging
import datasets

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM, AutoTokenizer, \
    BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq, set_seed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

import argparse
import bitsandbytes as bnb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tools.nltk_tools.nltk_tool import NLTKTool
from evaluation.baselines.prompt import get_prompt


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


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
    fine-tune the T5 model in supervised learning mode
    """
    # seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(read_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config)  # T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(read_model_path)

    logging.info('\n\n')
    logging.info('---------------------------model loaded-------------------------------\n')

    # dataset preprocessing
    max_sentence_length = 1024
    # set n_positions to max sentence length
    model.config.n_positions = max_sentence_length

    def preprocess_function(samples):
        paragraphs = samples['paragraph']  # list of strings
        procedure_tuples_texts = samples['procedure_tuples_text']  # list of strings

        prompt = get_prompt()

        input_texts = [prompt.format(paragraph) for index, paragraph in enumerate(paragraphs)]

        assert len(input_texts) == len(procedure_tuples_texts)

        model_inputs = tokenizer(
            input_texts,
            padding="longest",
            max_length=max_sentence_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            procedure_tuples_texts,
            padding="longest",
            max_length=max_sentence_length,
            truncation=True,
            return_tensors="pt",
        )
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    nltk_tool = NLTKTool()

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, device='cpu')

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, device='cpu')

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

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=8,
        lora_alpha=512,
        target_modules=['q', 'v'],
        lora_dropout=0.01,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config)

    # process dataset
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=list(train_dataset.features))
    dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=list(dev_dataset.features))
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=list(test_dataset.features))

    logging.info('\n\n')
    logging.info('---------------------------dataset loaded-------------------------------')

    args, _ = parse_arge()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_save_path,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        generation_max_length=1024,  # max_sentence_length
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=True,  # Use BF16 if available    args.bf16
        learning_rate=1e-4,
        num_train_epochs=10,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,

        # logging & evaluation strategies
        logging_dir=f"logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=999,
        load_best_model_at_end=False,  # load_best_model_at_end requires the save and eval strategy to match
        # push to hub parameters
        # report_to="tensorboard",
        # push_to_hub=True if args.repository_id else False,
        # hub_strategy="every_save",
        # hub_model_id=args.repository_id if args.repository_id else None,
        # hub_token=args.hf_token,
        # auto_find_batch_size=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
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
        # Save our tokenizer and model
        tokenizer.save_pretrained(model_save_path)
        # trainer.create_model_card()  # creates a model card in the model_save_path directory
        trainer.model.save_pretrained(model_save_path)
        logging.info('\n\n')
        logging.info('---------------------------Save tokenizer and final model----------------------------')

    # test
    predictions = trainer.predict(test_dataset=test_dataset)
    # print(predictions.predictions.shape, predictions.label_ids.shape)

    test_result = compute_metrics((predictions.predictions, predictions.label_ids))
    logging.info('\n\n')
    logging.info('---------------------------test result-------------------------------')
    logging.info(test_result)
    logging.info('\n\n')


def train_main(save_model):
    # logging
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    read_model_path = '/path/to/initial/flan-t5-xxl'

    model_save_path = '/path/to/save/fine-tuned/flan-t5-xxl'

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
