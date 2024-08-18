import os
import sys
from queue import Queue
from datasets import Dataset

from tools import util_box

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def trans2triples(data_item):
    step_nodes = data_item['step_nodes']
    data_nodes = data_item['data_nodes']
    text_nodes = data_item['text_nodes']

    # Logic branch number record, to distinguish different logic branches
    XOR_num = 0
    OR_num = 0
    AND_num = 0
    # Transform the logical nodes in step_nodes, add the logical branch number
    for index, node in enumerate(step_nodes):
        node_type = node['type']
        if node_type in ['XOR']:
            XOR_num += 1
            step_nodes[index]['num'] = XOR_num
        elif node_type in ['OR']:
            OR_num += 1
            step_nodes[index]['num'] = OR_num
        elif node_type in ['AND']:
            AND_num += 1
            step_nodes[index]['num'] = AND_num
        else:
            pass

    SequenceFlow = data_item['SequenceFlow']
    MessageFlow = data_item['MessageFlow']
    Association = data_item['Association']

    step_nodes_map = {}  # step_nodes mapping
    for node in step_nodes:
        step_nodes_map[node['resourceId']] = node

    data_nodes_map = {}  # data_nodes mapping
    for node in data_nodes:
        data_nodes_map[node['resourceId']] = node

    text_nodes_map = {}  # text_nodes mapping
    for node in text_nodes:
        text_nodes_map[node['resourceId']] = node

    # aggregate all artifacts
    artifacts_map = {}
    artifacts_map.update(data_nodes_map)
    artifacts_map.update(text_nodes_map)

    step_node2step_node = {}  # step_node to step_node mapping
    for sequence_flow in SequenceFlow:
        src = sequence_flow['src']
        tgt = sequence_flow['tgt']
        condition = sequence_flow['condition']  # condition text, may be empty, and only SequenceFlow has this attribute
        if src not in step_node2step_node.keys():
            step_node2step_node[src] = [[tgt, condition]]
        else:
            step_node2step_node[src].append([tgt, condition])

    message_map = {}
    for message_flow in MessageFlow:
        src = message_flow['src']
        tgt = message_flow['tgt']
        if src not in message_map.keys():
            message_map[src] = [tgt]
        else:
            message_map[src].append(tgt)

    step2artifacts = {}
    for association in Association:
        src = association['src']
        tgt = association['tgt']
        if src in step_nodes_map.keys():  # src is step_node
            if src not in step2artifacts.keys():
                step2artifacts[src] = [tgt]
            else:
                step2artifacts[src].append(tgt)

    step2artifacts_back = {}  # be pointed by which artifacts
    for association in Association:
        src = association['src']
        tgt = association['tgt']
        if src in artifacts_map.keys():  # src is artifact
            if tgt not in step2artifacts_back.keys():
                step2artifacts_back[tgt] = [src]
            else:
                step2artifacts_back[tgt].append(src)

    # bfs ------------------------------------------------------------------------------------------------------------

    # find start nodes
    start_nodes_ids = [node['resourceId'] for node in step_nodes if node['type'] == 'StartNode']

    all_triples = []
    for start_node_id in start_nodes_ids:
        triples = []  # current triples

        start_node = step_nodes_map[start_node_id]
        agent = start_node['agent']  # can be empty

        # avoid loop
        seen_node_ids = set()
        queue = Queue()
        queue.put(start_node_id)  # add start node to queue
        while not queue.empty():
            cur_node_id = queue.get()
            if cur_node_id in seen_node_ids:
                continue
            seen_node_ids.add(cur_node_id)

            cur_node = step_nodes_map[cur_node_id]
            cur_node_type = cur_node['type']
            cur_node_text = cur_node['NodeText']
            # cur_node_id may not be in step_node2step_node, because there may be no out edges
            if cur_node_id in step_node2step_node:
                tgt_edges = step_node2step_node[cur_node_id]  # [[tgt, condition], ...]
            else:
                tgt_edges = []

            # 1. step nodes
            if cur_node_type in ['StartNode']:
                cur_node_info = "Start({})".format(cur_node_text) if len(cur_node_text) > 0 else "Start"
            elif cur_node_type in ['Activity']:
                cur_node_info = "{}".format(cur_node_text)
            elif cur_node_type in ['XOR']:
                node_num = cur_node['num']
                if len(cur_node_text) > 0:
                    cur_node_info = "XOR{} ({})".format(node_num, cur_node_text)
                else:
                    cur_node_info = "XOR{}".format(node_num)
            elif cur_node_type in ['OR']:
                node_num = cur_node['num']
                if len(cur_node_text) > 0:
                    cur_node_info = "OR{} ({})".format(node_num, cur_node_text)
                else:
                    cur_node_info = "OR{}".format(node_num)
            elif cur_node_type in ['AND']:
                node_num = cur_node['num']
                cur_node_info = "AND{}".format(node_num)
            else:  # EndNode not deal with
                pass

            for tgt_edge in tgt_edges:
                tgt = tgt_edge[0]
                condition = tgt_edge[1]

                if len(condition) > 0:
                    edge_info = "-> ({})".format(condition)
                else:
                    edge_info = "->"

                tgt_node = step_nodes_map[tgt]
                tgt_node_type = tgt_node['type']
                tgt_node_text = tgt_node['NodeText']

                if tgt_node_type in ['StartNode']:
                    sys.exit('Error: StartNode cannot be tgt node')
                elif tgt_node_type in ['Activity']:
                    tgt_node_info = "{}".format(tgt_node_text)
                elif tgt_node_type in ['XOR']:
                    node_num = tgt_node['num']
                    if len(tgt_node_text) > 0:
                        tgt_node_info = "XOR{} ({})".format(node_num, tgt_node_text)
                    else:
                        tgt_node_info = "XOR{}".format(node_num)
                elif tgt_node_type in ['OR']:
                    node_num = tgt_node['num']
                    if len(tgt_node_text) > 0:
                        tgt_node_info = "OR{} ({})".format(node_num, tgt_node_text)
                    else:
                        tgt_node_info = "OR{}".format(node_num)
                elif tgt_node_type in ['AND']:
                    node_num = tgt_node['num']
                    tgt_node_info = "AND{}".format(node_num)
                else:  # EndNode
                    if len(tgt_node_text) > 0:
                        tgt_node_info = "End({})".format(tgt_node_text)
                    else:
                        tgt_node_info = "End"

                # edge_info
                # tgt_node_info
                triple_this = "{} {} {}".format(cur_node_info, edge_info, tgt_node_info)
                triples.append(triple_this)

            # put them all into the queue
            for tgt_edge in tgt_edges:
                tgt = tgt_edge[0]
                if tgt not in seen_node_ids:
                    queue.put(tgt)

            # 2. artifacts
            if cur_node_id in step2artifacts:
                tgt_artifacts = step2artifacts[cur_node_id]
                for tgt_artifact in tgt_artifacts:
                    tgt_artifact_node = artifacts_map[tgt_artifact]
                    tgt_artifact_node_type = tgt_artifact_node['type']
                    tgt_artifact_node_text = tgt_artifact_node['NodeText']

                    if tgt_artifact_node_type in ['DataObject']:
                        tgt_artifact_node_info = "DataObject({})".format(tgt_artifact_node_text)
                    elif tgt_artifact_node_type in ['TextAnnotation']:
                        tgt_artifact_node_info = "TextAnnotation({})".format(tgt_artifact_node_text)
                    else:
                        continue

                    triple_this = "{} -> {}".format(cur_node_info, tgt_artifact_node_info)
                    triples.append(triple_this)

            # 3. artifacts pointed to cur node
            if cur_node_id in step2artifacts_back:
                src_artifacts = step2artifacts_back[cur_node_id]
                for src_artifact in src_artifacts:
                    src_artifact_node = artifacts_map[src_artifact]
                    src_artifact_node_type = src_artifact_node['type']
                    src_artifact_node_text = src_artifact_node['NodeText']

                    if src_artifact_node_type in ['DataObject']:
                        src_artifact_node_info = "DataObject({})".format(src_artifact_node_text)
                    elif src_artifact_node_type in ['TextAnnotation']:
                        sys.exit('Error: TextAnnotation cannot be src artifact node')
                        # src_artifact_node_info = "TextAnnotation({})".format(src_artifact_node_text)
                    else:
                        continue

                    triple_this = "{} -> {}".format(src_artifact_node_info, cur_node_info)
                    triples.append(triple_this)

        flow = {
            'agent': agent,  # can be empty
            'triples': triples,
        }

        all_triples.append(flow)

    # to text
    all_paragraphs = []
    for flow_item in all_triples:
        agent = flow_item['agent']
        triples = flow_item['triples']

        if len(agent) > 0:
            agent_info = "For {}:".format(agent)
            triples = [agent_info] + triples

        paragraph_this = '\n'.join(triples)
        all_paragraphs.append(paragraph_this)

    target_paragraph = '\n\n'.join(all_paragraphs)

    return target_paragraph


def data_prepare_hf_format(data_list):
    # print(data_list[0].keys())

    processed_data_dict = {
        'paragraph': [],
        'procedure_tuples_text': [],
    }
    for index, item in enumerate(data_list):
        paragraph = item['paragraph']
        target_paragraph = trans2triples(item)

        processed_data_dict['paragraph'].append(paragraph)
        processed_data_dict['procedure_tuples_text'].append(target_paragraph)

    processed_data_dict = Dataset.from_dict(processed_data_dict)

    return processed_data_dict


def prepare_main_hf_format(write_data):
    read_dir = 'dataset/procedural_graph_extraction'
    read_train_path = os.path.join(read_dir, 'train.json')
    read_valid_path = os.path.join(read_dir, 'dev.json')
    read_test_path = os.path.join(read_dir, 'test.json')

    read_train_data = util_box.read_json(read_train_path)
    read_valid_data = util_box.read_json(read_valid_path)
    read_test_data = util_box.read_json(read_test_path)
    print('len of read_train_data: ', len(read_train_data))
    print('len of read_valid_data: ', len(read_valid_data))
    print('len of read_test_data: ', len(read_test_data))

    processed_train_data = data_prepare_hf_format(read_train_data)
    processed_dev_data = data_prepare_hf_format(read_valid_data)
    processed_test_data = data_prepare_hf_format(read_test_data)

    write_path = 'evaluation/baselines/training_data'

    if write_data:
        os.makedirs(os.path.join(write_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(write_path, 'dev'), exist_ok=True)
        os.makedirs(os.path.join(write_path, 'test'), exist_ok=True)
        processed_train_data.save_to_disk(os.path.join(write_path, 'train'))
        processed_dev_data.save_to_disk(os.path.join(write_path, 'dev'))
        processed_test_data.save_to_disk(os.path.join(write_path, 'test'))


if __name__ == '__main__':
    prepare_main_hf_format(write_data=True)
