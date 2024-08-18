import logging
import os
import sys

import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.eva_utils.eva_graph2triple import eva_trans2triples
from evaluation.metric.lexical_based import LexicalBased
from tools import util_box

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def single_eva(read_path, model_name, eva_type):
    metric = LexicalBased()

    metric.clear()

    read_data = util_box.read_json(read_path)

    eva_results = {
        'model_name': model_name,
        'eva_type': eva_type,
        'metric_name': "lexical_based",
        'results': []
    }

    logger.info("-----------------*** Evaluating ***-----------------")
    logger.info('model: ' + model_name + ', metric: lexical_based' + ', eva_type: ' + str(eva_type))
    logger.info('len of data to eva: ' + str(len(read_data['model_outputs'])))
    logger.info("-----------------*** Evaluating ***-----------------")

    # save_outputs = {
    #     'model_name': model_name,
    #     'eva_type': eva_type,
    #     'model_outputs': [],
    # }

    model_outputs = read_data['model_outputs']

    t_bar = tqdm.tqdm(model_outputs, desc='Evaluating')
    for index, sample in enumerate(t_bar):
        model_output = sample['model_output']  # model's output to be evaluated

        # compute metric
        all_triples, target_paragraph = eva_trans2triples(sample)

        sample_result = metric.update(model_output, all_triples)

        # save single eva result
        sample['target_paragraph'] = target_paragraph
        sample['eva_result'] = sample_result
        eva_results['results'].append(sample)

        # set t_bar postfix
        cur_results = metric.cur_results()  # dict
        t_bar.set_postfix(cur_results)

    # detailed results
    node_detailed_results, edge_detailed_results, node_total_results, edge_total_results, node_eva_results, other_results = metric.get_detailed_results()

    return eva_results, node_eva_results, edge_detailed_results, node_total_results, edge_total_results, other_results


def eva_choose_models(model_outputs_save_dir, eva_results_save_path, write_results, overwrite):
    eva_type = "test"  # default test

    metric_name = "lexical_based"  # default lexical_based

    eva_models = ["llama2_70b_few_shot"]

    file_save_path = os.path.join(eva_results_save_path, str(eva_type) + '_results_' + str(metric_name) + '.json')
    if os.path.exists(file_save_path):
        existed_results = util_box.read_json(file_save_path)
        logger.info("-----------------*** read already existed results ***-----------------\n\n")
    else:
        existed_results = {}

    # eva all models
    for model_name in eva_models:
        if model_name in existed_results.keys() and not overwrite:
            logger.info("-----------------*** already existed results ***-----------------")
            logger.info('model: ' + model_name + ', metric: ' + str(metric_name) + ', eva_type: ' + str(eva_type))
            logger.info("-----------------*** already existed results ***-----------------\n\n")
            continue

        read_path = os.path.join(model_outputs_save_dir, model_name + '_' + str(eva_type) + '.json')
        eva_results, node_eva_results, edge_detailed_results, node_total_results, edge_total_results, other_results = \
            single_eva(read_path, model_name, eva_type)

        if write_results:
            # node
            existed_results[model_name] = node_eva_results
            # edge
            for key in edge_detailed_results.keys():
                existed_results[model_name][key] = edge_detailed_results[key]
            # overall
            existed_results[model_name]['node_total'] = node_total_results
            existed_results[model_name]['edge_total'] = edge_total_results

            # agent
            existed_results[model_name]['agent_total'] = other_results['agent']['total']

            util_box.write_json(file_save_path, existed_results)
            logger.info("-----------------*** save results for model: {} ***-----------------\n\n\n".format(model_name))


def eva_single_model(model_outputs_save_dir, eva_results_save_path, save_eva_results):
    eva_type = "test"  # default test

    # choose model -----------------------------------------------------------------------------------
    model_name = "llama2_70b_few_shot"

    read_path = os.path.join(model_outputs_save_dir, model_name + '_' + str(eva_type) + '.json')
    eva_results, node_eva_results, edge_detailed_results, node_total_results, edge_total_results, other_results = \
        single_eva(read_path, model_name, eva_type)

    # save eva_results
    if save_eva_results:
        every_save_path = os.path.join(eva_results_save_path, model_name + '_' + str(eva_type) + '_' + "lexical_based" + '.json')
        util_box.write_json(every_save_path, eva_results)


def evaluation_main():
    model_outputs_save_dir = "evaluation/model_outputs_cache"  # output of models

    eva_results_save_path = "evaluation/eva_results"  # path to save eva results

    # # eva single model
    # eva_single_model(model_outputs_save_dir, eva_results_save_path, save_eva_results=True)

    # eva chosen models
    eva_choose_models(model_outputs_save_dir, eva_results_save_path, write_results=True, overwrite=False)


if __name__ == '__main__':
    evaluation_main()
