import logging
import os
import sys
import torch
import tqdm
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import util_box

from evaluation.baselines.ChatGPT.chatgpt_few_shot import PEChatGPTFew
from evaluation.baselines.FlanT5.fine_tune.flan_t5_xxl_for_eva import PEFlanT5LoraXXL
from evaluation.baselines.FlanT5.flan_t5_xxl_for_eva import PEFlanT5XXL
from evaluation.baselines.Llama2.Llama_2_70b_few import PELlama2For70BFew
from evaluation.baselines.Llama2.fine_tune.Llama_2_70b import PELlama2Lora70B

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_name, cuda_ip):
    # LM---------------------------------------------------------------------------------------------------

    if model_name == 'llama2_70b_few_shot':
        return PELlama2For70BFew(cuda_ip)
    elif model_name == 'llama2_70b_fine_tuned':
        return PELlama2Lora70B(cuda_ip)

    elif model_name == 'flan_t5_xxl_few_shot':
        return PEFlanT5XXL(cuda_ip)
    elif model_name == 'flan_t5_xxl_fine_tuned':
        return PEFlanT5LoraXXL(cuda_ip)

    elif model_name == 'chat_gpt_few_shot':
        return PEChatGPTFew()

    else:
        sys.exit('model_name error: ' + model_name)


def save_results(save_outputs, save_path):
    util_box.write_json(save_path, save_outputs)
    logger.info("-----------------*** save outputs ***-----------------")
    logger.info('save outputs: ' + str(len(save_outputs['model_outputs'])))
    logger.info("-----------------*** save outputs ***-----------------\n\n")


def single_eva(dataset, model_name, batch_size, save_path, eva_type, cuda_ip):
    model = load_model(model_name, cuda_ip)

    if os.path.exists(save_path):
        save_outputs = util_box.read_json(save_path)
        logger.info("-----------------*** load saved outputs ***-----------------")
        logger.info(save_path)
        logger.info('current model_outputs len: ' + str(len(save_outputs['model_outputs'])))
        logger.info("-----------------*** load saved outputs ***-----------------\n\n")
    else:
        save_outputs = {
            'model_name': model_name,
            'eva_type': eva_type,
            'model_outputs': [],
        }  # save the model outputs
        logger.info("-----------------*** no saved outputs ***-----------------")
        logger.info(save_path + " not exists!!!")
        logger.info("prepare empty save_outputs to save")
        logger.info("-----------------*** no saved outputs ***-----------------\n\n")

    logger.info("-----------------*** Save For Evaluating ***-----------------")
    logger.info('model_name: ' + model_name + ', eva_type: ' + str(eva_type) + ', cuda_ip: ' + str(cuda_ip))
    logger.info("-----------------*** Save For Evaluating ***-----------------\n\n")

    already_eva_file_index = [item['file_index'] for item in save_outputs['model_outputs']]

    rest_dataset = []
    for sample in dataset:
        file_index = sample['file_index']
        if file_index not in already_eva_file_index:
            rest_dataset.append(sample)

    # save model outputs
    t_bar = tqdm.tqdm(total=len(rest_dataset), desc='Save Evaluating')
    for index in range(0, len(rest_dataset), batch_size):

        samples = rest_dataset[index: index + batch_size]

        # generate
        paragraphs = [sample['paragraph'] for sample in samples]
        batch_outputs = model.batch_predict(paragraphs)

        assert len(batch_outputs) == len(samples)

        # save model outputs
        for i, output in enumerate(batch_outputs):
            save_sample = samples[i]
            save_sample['model_output'] = str(output)

            save_outputs['model_outputs'].append(save_sample)

        t_bar.update(len(samples))
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        t_bar.set_postfix(time=time_str)

        # save to desk
        save_results(save_outputs, save_path)

    logger.info("-----------------*** save finished ***-----------------")
    logger.info("save_path: " + save_path)
    logger.info("-----------------*** save finished ***-----------------\n\n")


def save_single_model(dataset, model_outputs_save_dir, eva_type, overwrite_outputs):  # overwrite_outputs太危险了 不用
    model_name = "llama2_70b_few_shot"

    save_path = os.path.join(model_outputs_save_dir, model_name + '_' + str(eva_type) + '.json')

    batch_size = 4
    cuda_ip = "0,1,2,3"

    single_eva(dataset, model_name, batch_size, save_path, eva_type, cuda_ip=cuda_ip)


def save_for_evaluation_main():
    dataset_dir = "dataset/procedural_graph_extraction"
    eva_type = "test"
    dataset_path = os.path.join(dataset_dir, eva_type + ".json")
    dataset = util_box.read_json(dataset_path)
    # ---------------------------------------read datasets-------------------------------------------------

    model_outputs_save_dir = "evaluation/model_outputs_cache"
    # ---------------------------------------save model dir---------------------------------------------

    save_single_model(dataset, model_outputs_save_dir, eva_type, overwrite_outputs=False)


if __name__ == '__main__':
    save_for_evaluation_main()
