import sys
import spacy
import re
from nltk.translate.bleu_score import sentence_bleu


def get_gateway_content(text, tag):
    text = re.sub(r'{}(\d+)'.format(tag), '', text)
    text = text.strip()

    if not re.search(r'\((.*?)\)', text):
        return ""
    result = re.search(r'\((.*?)\)', text).group(1)
    result = result.strip()
    return result


def parse_model_output(model_output, split_tag):
    all_parsed_spans = []
    cur_agent = ''  # record the current agent
    for line in model_output.split(split_tag):
        line = line.strip()
        # Node -------------------------------------------------------------------------------------------------
        if "->" in line:  # there is an arrow, which means it is a node + edge
            all_parsed_spans.append([line, cur_agent])
        # Agent --------------------------------------------------------------------------------------------------
        elif line.startswith('For') or line.startswith('for'):
            if line.startswith('For'):
                agent = re.sub(r'^For\s+', '', line)
            else:
                agent = re.sub(r'^for\s+', '', line)
            # delete the colon at the end
            if agent.endswith(':'):
                agent = agent[:-1]
            agent = agent.strip()
            cur_agent = agent
        # Other --------------------------------------------------------------------------------------------------
        else:  # nothing
            pass
    return all_parsed_spans


def lexical_score(cur_span, refer_span, spacy_tool, args):
    if len(cur_span) == 0 or len(refer_span) == 0:
        return 0

    cur_lemma = spacy_tool.get_lemma(cur_span)  # tokenization
    refer_lemma = spacy_tool.get_lemma(refer_span)

    # compare lemma
    p_t = len(cur_lemma)
    r_t = len(refer_lemma)
    p_r = 0
    r_r = 0
    for lemma in cur_lemma:
        if lemma in refer_lemma:
            p_r += 1
    for lemma in refer_lemma:
        if lemma in cur_lemma:
            r_r += 1
    p_s = p_r / p_t if p_t > 0 else 0
    r_s = r_r / r_t if r_t > 0 else 0
    f_s = (2 * p_s * r_s) / (p_s + r_s) if (p_s + r_s) > 0 else 0

    return f_s


def parse_node_type(node_span):
    node_span = node_span.strip()

    if len(node_span) == 0:
        return False, None, None

    # Start: Start(content) / Start
    # End: End(content) / End
    # Activity: Activity
    # XOR: XOR1 / XOR1 (content)
    # OR: OR1 / OR1 (content)
    # AND: AND1
    # DataObject: DataObject(content)
    # TextAnnotation: TextAnnotation(content)

    # 'resourceId': node_id,
    # 'NodeText': node_text,
    # 'agent': '',
    # 'type': node_type,
    # })

    # 1. DataObject
    if "DataObject" in node_span:
        part_re = re.search(r'\(.+\)', node_span)
        if part_re:
            part_span = part_re.span()
            node_text = node_span[part_span[0] + 1: part_span[1] - 1]
            node_text = node_text.strip()
            if len(node_text) == 0:
                return False, None, None
            node_type = 'DataObject'
        else:
            return False, None, None
    # 2. TextAnnotation
    elif "TextAnnotation" in node_span:
        part_re = re.search(r'\(.+\)', node_span)
        if part_re:
            part_span = part_re.span()
            node_text = node_span[part_span[0] + 1: part_span[1] - 1]
            node_text = node_text.strip()
            if len(node_text) == 0:
                return False, None, None
            node_type = 'TextAnnotation'
        else:
            return False, None, None

    # 3. Start
    elif node_span.startswith("Start") or node_span.startswith("start"):
        part_re = re.search(r'\(.+\)', node_span)
        if part_re:
            part_span = part_re.span()
            node_text = node_span[part_span[0] + 1: part_span[1] - 1]
            node_text = node_text.strip()
            node_type = 'StartNode'
        else:
            node_text = ""
            node_type = 'StartNode'
    # 4. End
    elif node_span.startswith("End") or node_span.startswith("end"):
        part_re = re.search(r'\(.+\)', node_span)
        if part_re:
            part_span = part_re.span()
            node_text = node_span[part_span[0] + 1: part_span[1] - 1]
            node_text = node_text.strip()
            node_type = 'EndNode'
        else:
            node_text = ""
            node_type = 'EndNode'

    # 5. XOR
    elif re.match(r'^XOR\d*', node_span) or re.match(r'^xor\d+', node_span):
        node_text = get_gateway_content(node_span, 'XOR')
        node_type = 'XOR'

    # 6. OR
    elif re.match(r'^OR\d*', node_span) or re.match(r'^or\d+', node_span):
        node_text = get_gateway_content(node_span, 'OR')
        node_type = 'OR'

    # 7. AND
    elif re.match(r'^AND\d*', node_span) or re.match(r'^and\d+', node_span):
        node_text = get_gateway_content(node_span, 'AND')
        node_type = 'AND'

    # 8. Activity
    else:
        if len(node_span) > 0:
            node_text = node_span
            node_text = node_text.strip()
            node_type = 'Activity'
        else:
            return False, None, None

    return True, node_text, node_type


def change_node_match_flag(all_parsed_nodes, node_id, match_flag):
    if node_id is None:
        sys.exit('node_id is None!!!')

    for index, item in enumerate(all_parsed_nodes):
        if item['resourceId'] == node_id:
            all_parsed_nodes[index]['match'] = match_flag
            break
    return all_parsed_nodes


def same_parsed_match(cur_node_text, refer_node_text, spacy_tool, args):
    # avoid the influence of count an already matched node
    if len(cur_node_text) == 0 and len(refer_node_text) == 0:
        return True
    elif len(cur_node_text) == 0 or len(refer_node_text) == 0:
        return False
    else:
        bleu_score = lexical_score(cur_node_text, refer_node_text, spacy_tool, args)
        if bleu_score >= args['same_parsed_match_threshold']:
            return True
        else:
            return False


def find_node(all_parsed_nodes, node_text, cur_agent, node_type, source_span, spacy_tool, args):
    # check if the node has been parsed
    find_flag = False
    match_flag = False
    node_id = None
    for item in all_parsed_nodes:
        if (item['type'] == node_type) and (item['agent'] == cur_agent):  # (item['agent'] == cur_agent)
            if node_type in ['XOR', 'OR', 'AND']:
                if source_span == item['source_span']:
                    find_flag = True
                    match_flag = item['match']
                    node_id = item['resourceId']
                    break
                else:
                    pass

            else:
                if same_parsed_match(node_text, item['NodeText'], spacy_tool, args):
                    find_flag = True
                    match_flag = item['match']
                    node_id = item['resourceId']
                    break
    return find_flag, match_flag, node_id


def add_node(all_parsed_nodes, node_text, cur_agent, node_type, source_span, match_flag):
    # add the node
    id_for_add_node = len(all_parsed_nodes)  # start from 0
    all_parsed_nodes.append({
        'resourceId': id_for_add_node,
        'NodeText': node_text,
        'agent': cur_agent,
        'type': node_type,
        'match': match_flag,
        'source_span': source_span,
    })

    return all_parsed_nodes, id_for_add_node


def parse_span(line, agent, spacy_tool, args):
    if "->" not in line:  # there must be an arrow, which means it is a node + edge
        return False, None, None

    # parse edge information first
    condition = ""
    flag = re.search(r'->\s*\(.+\).', line)
    if flag:
        _span = flag.span()
        part_text = line[_span[0]: _span[1]]
        part_span = re.search(r'\(.+\)', part_text).span()
        edge_content = part_text[part_span[0] + 1: part_span[1] - 1]

        edge_type = 'SequenceFlow'
        condition = edge_content

        # remove the content in the brackets
        line = line.replace('(' + edge_content + ')', '')
    else:
        # 3.  ->
        edge_type = 'SequenceFlow'

    # then parse node information
    node_spans = line.split('->')
    if len(node_spans) == 2:
        node_spans = [span.strip() for span in node_spans]
        cur_node_1 = None
        cur_node_2 = None
        is_node, node_text, node_type = parse_node_type(node_spans[0])
        if is_node:
            if node_type in ['DataObject', 'TextAnnotation']:
                edge_type = 'Association'
            # add node information
            cur_node_1 = {
                'NodeText': node_text,
                'agent': agent,
                'type': node_type,
                'source_span': node_spans[0].strip(),
            }

        is_node, node_text, node_type = parse_node_type(node_spans[1])
        if is_node:
            if node_type in ['DataObject', 'TextAnnotation']:
                edge_type = 'Association'
            # add node information
            cur_node_2 = {
                'NodeText': node_text,
                'agent': agent,
                'type': node_type,
                'source_span': node_spans[0].strip(),
            }
        # edge information
        edge_info = [edge_type, condition]
        node_info = [cur_node_1, cur_node_2]
        return True, edge_info, node_info
    else:
        return False, None, None


def node_match(cur_node_item, refer_node_item, spacy_tool, args):
    if not cur_node_item['type'] == refer_node_item['type']:
        return False

    if cur_node_item['type'] in ['StartNode', 'EndNode']:
        return True

    if cur_node_item['type'] in ['OR']:
        return True

    # type matched, then check if the text matched
    if len(cur_node_item['NodeText']) == 0 and len(refer_node_item['NodeText']) == 0:
        return True
    elif len(cur_node_item['NodeText']) == 0 or len(refer_node_item['NodeText']) == 0:
        return False
    else:
        bleu_score = lexical_score(cur_node_item['NodeText'], refer_node_item['NodeText'], spacy_tool, args)
        # print(cur_node_item['NodeText'] + ' --- ' + refer_node_item['NodeText'] + ' --- ' + str(bleu_score))
        if bleu_score >= args['node_match_threshold']:
            return True
        else:
            return False


def condition_match(cur_condition, refer_condition, spacy_tool, args):
    if len(cur_condition) == 0 and len(refer_condition) == 0:
        return 1
    elif len(cur_condition) == 0 or len(refer_condition) == 0:
        return 0
    else:
        bleu_score = lexical_score(cur_condition, refer_condition, spacy_tool, args)
        return bleu_score


def parse_span_match(cur_span_item, refer_span_item, exist_nodes, spacy_tool, args):
    node_parse_num = 0
    node_match_num = 0
    edge_parse_num = 0
    edge_match_num = 0

    # node
    node_types = ['Activity', 'StartNode', 'EndNode', 'XOR', 'OR', 'AND', 'DataObject', 'TextAnnotation']
    node_statistics = {}
    for node_type in node_types:
        node_statistics[node_type] = {
            'total': 0,
            'correct': 0,
        }

    # edge
    edge_types = ['SequenceFlow', 'Association']
    edge_statistics = {}
    for edge_type in edge_types:
        edge_statistics[edge_type] = {
            'total': 0,
            'correct': 0,
        }

    # agent
    agent_types = ['Activity', 'StartNode', 'EndNode']
    agent_statistics = {}
    for agent_type in agent_types:
        agent_statistics[agent_type] = {
            'total': 0,
            'correct': 0,
        }

    # BLEU based -----
    # node
    bleu_node_types = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
    bleu_node_statistics = {}
    for node_type in bleu_node_types:
        bleu_node_statistics[node_type] = {
            'bleu_total': 0,
            'bleu_num': 0,
        }

    # agent
    bleu_agent_types = ['Activity', 'StartNode', 'EndNode']
    bleu_agent_statistics = {}
    for agent_type in bleu_agent_types:
        bleu_agent_statistics[agent_type] = {
            'bleu_total': 0,
            'bleu_num': 0,
        }

    # edge
    bleu_edge_types = ['SequenceFlow', 'Association']
    bleu_edge_statistics = {}
    for edge_type in bleu_edge_types:
        bleu_edge_statistics[edge_type] = {
            'total': 0,
            'correct': 0,
        }
    # ConditionFlow
    bleu_edge_statistics['ConditionFlow'] = {
        'bleu_total': 0,
        'bleu_num': 0,
    }

    cur_success_flag, cur_edge_info, cur_node_info = parse_span(cur_span_item[0], cur_span_item[1], spacy_tool, args)
    refer_success_flag, refer_edge_info, refer_node_info = parse_span(refer_span_item[0], refer_span_item[1], spacy_tool, args)

    # match
    if cur_success_flag:  # only when the model has made a prediction can it be counted
        # first check if the start node has been parsed
        if cur_node_info[0] is not None:
            need_match = False  # no need to match if the node has been matched or the refer node was not parsed successfully
            start_node_find, start_node_match, start_node_id = find_node(exist_nodes, cur_node_info[0]['NodeText'], cur_node_info[0]['agent'], cur_node_info[0]['type'], cur_node_info[0]['source_span'], spacy_tool, args)
            if not start_node_find:
                # if the node has been parsed successfully and has not been seen before, it needs to be added to the list of nodes, and then it will be known whether the node has been parsed before
                exist_nodes, start_node_id = add_node(exist_nodes, cur_node_info[0]['NodeText'], cur_node_info[0]['agent'], cur_node_info[0]['type'], cur_node_info[0]['source_span'], False)
                need_match = True
                node_parse_num += 1
                node_statistics[cur_node_info[0]['type']]['total'] += 1
                if cur_node_info[0]['type'] in agent_types and len(cur_node_info[0]['agent']) > 0:  # with agent
                    agent_statistics[cur_node_info[0]['type']]['total'] += 1
                    bleu_agent_statistics[cur_node_info[0]['type']]['bleu_num'] += 1

                # BLEU based: only count: ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
                bleu_nodes = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
                if cur_node_info[0]['type'] in bleu_nodes:
                    bleu_node_statistics[cur_node_info[0]['type']]['bleu_num'] += 1

            else:
                if not start_node_match:
                    need_match = True

            if need_match and refer_success_flag and (refer_node_info[0] is not None):
                # need_match and refer_node_info[0] is not None
                now_match = node_match(cur_node_info[0], refer_node_info[0], spacy_tool, args)
                if now_match:
                    node_match_num += 1
                    node_statistics[cur_node_info[0]['type']]['correct'] += 1
                    exist_nodes = change_node_match_flag(exist_nodes, start_node_id, True)
                    # agent match
                    if cur_node_info[0]['type'] in agent_types and len(cur_node_info[0]['agent']) > 0:  # with agent
                        agent_score = lexical_score(cur_node_info[0]['agent'], refer_node_info[0]['agent'], spacy_tool, args)
                        if agent_score >= args['agent_match_threshold']:
                            agent_statistics[cur_node_info[0]['type']]['correct'] += 1

                # BLEU based
                bleu_nodes = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
                if cur_node_info[0]['type'] in bleu_nodes:
                    # if the type is matched
                    if not cur_node_info[0]['type'] == refer_node_info[0]['type']:
                        bleu_score = 0
                    else:
                        bleu_score = spacy_tool.compute_bleu(cur_node_info[0]['NodeText'], refer_node_info[0]['NodeText'])
                        if cur_node_info[0]['type'] in ['StartNode', 'EndNode']:
                            bleu_score = 1  # StartNode and EndNode do not need to match text
                        # agent
                        if cur_node_info[0]['type'] in bleu_agent_types and len(cur_node_info[0]['agent']) > 0:
                            agent_bleu = spacy_tool.compute_bleu(cur_node_info[0]['agent'], refer_node_info[0]['agent'])
                            bleu_agent_statistics[cur_node_info[0]['type']]['bleu_total'] += agent_bleu

                    bleu_node_statistics[cur_node_info[0]['type']]['bleu_total'] += bleu_score
                    start_node_bleu_score = bleu_score
                else:
                    if cur_node_info[0]['type'] == refer_node_info[0]['type']:
                        start_node_bleu_score = 1

        # tail node
        if cur_node_info[1] is not None:
            need_match = False
            end_node_find, end_node_match, end_node_id = find_node(exist_nodes, cur_node_info[1]['NodeText'], cur_node_info[1]['agent'], cur_node_info[1]['type'], cur_node_info[1]['source_span'], spacy_tool, args)
            if not end_node_find:
                exist_nodes, end_node_id = add_node(exist_nodes, cur_node_info[1]['NodeText'], cur_node_info[1]['agent'], cur_node_info[1]['type'], cur_node_info[1]['source_span'], False)
                need_match = True
                node_parse_num += 1
                node_statistics[cur_node_info[1]['type']]['total'] += 1
                if cur_node_info[1]['type'] in agent_types and len(cur_node_info[1]['agent']) > 0:
                    agent_statistics[cur_node_info[1]['type']]['total'] += 1
                    bleu_agent_statistics[cur_node_info[1]['type']]['bleu_num'] += 1

                bleu_nodes = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
                if cur_node_info[1]['type'] in bleu_nodes:
                    bleu_node_statistics[cur_node_info[1]['type']]['bleu_num'] += 1
            else:
                if not end_node_match:
                    need_match = True

            if need_match and refer_success_flag and (refer_node_info[1] is not None):
                now_match = node_match(cur_node_info[1], refer_node_info[1], spacy_tool, args)
                if now_match:
                    node_match_num += 1
                    node_statistics[cur_node_info[1]['type']]['correct'] += 1
                    exist_nodes = change_node_match_flag(exist_nodes, end_node_id, True)
                    # agent match
                    if cur_node_info[1]['type'] in agent_types and len(cur_node_info[1]['agent']) > 0:
                        agent_score = lexical_score(cur_node_info[1]['agent'], refer_node_info[1]['agent'], spacy_tool, args)
                        if agent_score >= args['agent_match_threshold']:
                            agent_statistics[cur_node_info[1]['type']]['correct'] += 1

                # BLEU based
                bleu_nodes = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
                if cur_node_info[1]['type'] in bleu_nodes:
                    if not cur_node_info[1]['type'] == refer_node_info[1]['type']:
                        bleu_score = 0
                    else:
                        bleu_score = spacy_tool.compute_bleu(cur_node_info[1]['NodeText'], refer_node_info[1]['NodeText'])
                        if cur_node_info[1]['type'] in ['StartNode', 'EndNode']:
                            bleu_score = 1

                        if cur_node_info[1]['type'] in bleu_agent_types and len(cur_node_info[1]['agent']) > 0:
                            agent_bleu = spacy_tool.compute_bleu(cur_node_info[1]['agent'], refer_node_info[1]['agent'])
                            bleu_agent_statistics[cur_node_info[1]['type']]['bleu_total'] += agent_bleu

                    bleu_node_statistics[cur_node_info[1]['type']]['bleu_total'] += bleu_score
                    end_node_bleu_score = bleu_score
                else:
                    if cur_node_info[1]['type'] == refer_node_info[1]['type']:
                        end_node_bleu_score = 1

        # edge
        edge_parse_num += 1
        edge_statistics[cur_edge_info[0]]['total'] += 1
        edge_type = cur_edge_info[0]
        if edge_type == 'SequenceFlow':
            if len(cur_edge_info[1]) > 0:
                edge_type = 'ConditionFlow'

        if edge_type in ['SequenceFlow', 'Association']:
            bleu_edge_statistics[edge_type]['total'] += 1
        elif edge_type in ['ConditionFlow']:
            bleu_edge_statistics[edge_type]['bleu_num'] += 1
        else:
            sys.exit('edge_type error!!!')

        # both start and tail nodes need to be parsed to match the edge

        if cur_success_flag and refer_success_flag and (cur_node_info[0] is not None) and (cur_node_info[1] is not None) and (refer_node_info[0] is not None) and (refer_node_info[1] is not None):
            if node_match(cur_node_info[0], refer_node_info[0], spacy_tool, args) and node_match(cur_node_info[1], refer_node_info[1], spacy_tool, args):
                if cur_edge_info[0] == refer_edge_info[0]:  # the type of the edge matches
                    # the condition of the edge matches, or no condition at all
                    condition_score = condition_match(cur_edge_info[1], refer_edge_info[1], spacy_tool, args)
                    if condition_score >= args['condition_match_threshold']:
                        edge_match_num += 1
                        edge_statistics[cur_edge_info[0]]['correct'] += 1

            # BLEU score for edge
            if (cur_node_info[0]['type'] == refer_node_info[0]['type']) and (cur_node_info[1]['type'] == refer_node_info[1]['type']):
                if cur_edge_info[0] == refer_edge_info[0]:  # the type of the edge matches
                    # start
                    if cur_node_info[0]['type'] in ['StartNode', 'EndNode', 'AND']:
                        start_bleu = 1
                    else:
                        if cur_node_info[0]['type'] in ['XOR', 'OR']:
                            if len(cur_node_info[0]['NodeText']) == 0 and len(refer_node_info[0]['NodeText']) == 0:
                                start_bleu = 1
                            else:
                                start_bleu = spacy_tool.compute_bleu(cur_node_info[0]['NodeText'], refer_node_info[0]['NodeText'])
                        else:
                            start_bleu = spacy_tool.compute_bleu(cur_node_info[0]['NodeText'], refer_node_info[0]['NodeText'])
                    # end
                    if cur_node_info[1]['type'] in ['StartNode', 'EndNode', 'AND']:
                        end_bleu = 1
                    else:
                        if cur_node_info[1]['type'] in ['XOR', 'OR']:
                            if len(cur_node_info[1]['NodeText']) == 0 and len(refer_node_info[1]['NodeText']) == 0:
                                end_bleu = 1
                            else:
                                end_bleu = spacy_tool.compute_bleu(cur_node_info[1]['NodeText'], refer_node_info[1]['NodeText'])
                        else:
                            end_bleu = spacy_tool.compute_bleu(cur_node_info[1]['NodeText'], refer_node_info[1]['NodeText'])

                    # for different edge types
                    if edge_type in ['SequenceFlow', 'Association']:
                        if start_bleu >= 0.5 and end_bleu >= 0.5:
                            bleu_edge_statistics[edge_type]['correct'] += 1
                    elif edge_type in ['ConditionFlow']:
                        if start_bleu >= 0.5 and end_bleu >= 0.5:
                            # compute BLEU
                            assert len(cur_edge_info[1]) > 0
                            condition_bleu_score = spacy_tool.compute_bleu(cur_edge_info[1], refer_edge_info[1])
                            # print('cur:', cur_edge_info[1], ' --- ', 'refer:', refer_edge_info[1], 'score:', condition_bleu_score)
                            bleu_edge_statistics[edge_type]['bleu_total'] += condition_bleu_score
                    else:
                        sys.exit('edge_type error!!!')

    assert edge_parse_num >= edge_match_num

    return exist_nodes, node_parse_num, node_match_num, edge_parse_num, edge_match_num, node_statistics, edge_statistics, agent_statistics, bleu_node_statistics, bleu_agent_statistics, bleu_edge_statistics


def single_span_match(cur_span_item, refer_spans, exist_nodes, spacy_tool, args):
    # 1. find the most similar span
    match_index = -1
    max_bleu_score = -1
    for i, refer_span_item in enumerate(refer_spans):
        cur_score = lexical_score(cur_span_item[0], refer_span_item[0], spacy_tool, args)
        if cur_score > max_bleu_score:
            max_bleu_score = cur_score
            match_index = i

    # 2. check if can match
    if max_bleu_score >= args['span_threshold']:
        match_refer_span_item = refer_spans[match_index]
        # 3. match

        # print('cur_span: ', cur_span_item[0])
        # print('refer_span: ', match_refer_span_item[0])

        exist_nodes, node_parse_num, node_match_num, edge_parse_num, edge_match_num, node_statistics, edge_statistics, agent_statistics, bleu_node_statistics, bleu_agent_statistics, bleu_edge_statistics \
            = parse_span_match(cur_span_item, match_refer_span_item, exist_nodes, spacy_tool, args)
        # print(str(node_parse_num) + ' --- ' + str(node_match_num))
        # print('----------------------------\n')

    else:
        return False, exist_nodes, 0, 0, 0, 0, None, None, None, None, None, None

    return True, exist_nodes, node_parse_num, node_match_num, edge_parse_num, edge_match_num, node_statistics, edge_statistics, agent_statistics, bleu_node_statistics, bleu_agent_statistics, bleu_edge_statistics


def spans_exact_match(all_parsed_spans, all_triples, spacy_tool, args):
    # all_parsed_spans list of [span, agent]

    refer_spans = []
    for flow in all_triples:
        agent = flow['agent']
        triples = flow['triples']
        spans_this = [[span, agent] for span in triples]
        refer_spans.extend(spans_this)
    # refer_spans list of [span, agent]

    node_precision_total = 0
    node_precision_correct = 0
    node_recall_total = 0
    node_recall_correct = 0

    edge_precision_total = 0
    edge_precision_correct = 0
    edge_recall_total = 0
    edge_recall_correct = 0

    # node
    node_types = ['Activity', 'StartNode', 'EndNode', 'XOR', 'OR', 'AND', 'DataObject', 'TextAnnotation']
    node_statistics = {}
    for node_type in node_types:
        node_statistics[node_type] = {
            'p_total': 0,
            'p_correct': 0,
            'r_total': 0,
            'r_correct': 0,
        }

    # edge
    edge_types = ['SequenceFlow', 'Association']
    edge_statistics = {}
    for edge_type in edge_types:
        edge_statistics[edge_type] = {
            'p_total': 0,
            'p_correct': 0,
            'r_total': 0,
            'r_correct': 0,
        }

    # agent
    agent_types = ['Activity', 'StartNode', 'EndNode']  # 需要agent的节点类型
    agent_statistics = {}
    for agent_type in agent_types:
        agent_statistics[agent_type] = {
            'p_total': 0,
            'p_correct': 0,
            'r_total': 0,
            'r_correct': 0,
        }

    # BLEU based -----
    # node
    bleu_node_types = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
    bleu_node_statistics = {}
    for node_type in bleu_node_types:
        bleu_node_statistics[node_type] = {
            'bleu_precision_total': 0,
            'bleu_precision_num': 0,
            'bleu_recall_total': 0,
            'bleu_recall_num': 0,
        }

    # agent
    bleu_agent_types = ['Activity', 'StartNode', 'EndNode']
    bleu_agent_statistics = {}
    for agent_type in bleu_agent_types:
        bleu_agent_statistics[agent_type] = {
            'bleu_precision_total': 0,
            'bleu_precision_num': 0,
            'bleu_recall_total': 0,
            'bleu_recall_num': 0,
        }

    # edge
    bleu_edge_types = ['SequenceFlow', 'Association']
    bleu_edge_statistics = {}
    for edge_type in bleu_edge_types:
        bleu_edge_statistics[edge_type] = {
            'p_total': 0,
            'p_correct': 0,
            'r_total': 0,
            'r_correct': 0,
        }
    # ConditionFlow
    bleu_edge_statistics['ConditionFlow'] = {
        'bleu_precision_total': 0,
        'bleu_precision_num': 0,
        'bleu_recall_total': 0,
        'bleu_recall_num': 0,
    }

    # precision
    pred_nodes = []  # record the nodes parsed by the prediction to avoid calculating the same node repeatedly
    for parsed_span_item in all_parsed_spans:
        matched, pred_nodes, node_parse_num, node_match_num, edge_parse_num, edge_match_num, p_node_statistics, p_edge_statistics, p_agent_statistics, p_bleu_node_statistics, p_bleu_agent_statistics, p_bleu_edge_statistics = single_span_match(parsed_span_item, refer_spans, pred_nodes, spacy_tool, args)
        if matched:
            node_precision_total += node_parse_num
            node_precision_correct += node_match_num
            edge_precision_total += edge_parse_num
            edge_precision_correct += edge_match_num

            for node_type in node_types:
                node_statistics[node_type]['p_total'] += p_node_statistics[node_type]['total']
                node_statistics[node_type]['p_correct'] += p_node_statistics[node_type]['correct']
            for edge_type in edge_types:
                edge_statistics[edge_type]['p_total'] += p_edge_statistics[edge_type]['total']
                edge_statistics[edge_type]['p_correct'] += p_edge_statistics[edge_type]['correct']

            for agent_type in agent_types:
                agent_statistics[agent_type]['p_total'] += p_agent_statistics[agent_type]['total']
                agent_statistics[agent_type]['p_correct'] += p_agent_statistics[agent_type]['correct']

            # print(p_bleu_edge_statistics)
            # BLEU
            for node_type in bleu_node_types:
                bleu_node_statistics[node_type]['bleu_precision_total'] += p_bleu_node_statistics[node_type]['bleu_total']
                bleu_node_statistics[node_type]['bleu_precision_num'] += p_bleu_node_statistics[node_type]['bleu_num']
            for agent_type in bleu_agent_types:
                bleu_agent_statistics[agent_type]['bleu_precision_total'] += p_bleu_agent_statistics[agent_type]['bleu_total']
                bleu_agent_statistics[agent_type]['bleu_precision_num'] += p_bleu_agent_statistics[agent_type]['bleu_num']
            for edge_type in bleu_edge_types:
                bleu_edge_statistics[edge_type]['p_total'] += p_bleu_edge_statistics[edge_type]['total']
                bleu_edge_statistics[edge_type]['p_correct'] += p_bleu_edge_statistics[edge_type]['correct']
            # ConditionFlow
            bleu_edge_statistics['ConditionFlow']['bleu_precision_total'] += p_bleu_edge_statistics['ConditionFlow']['bleu_total']
            bleu_edge_statistics['ConditionFlow']['bleu_precision_num'] += p_bleu_edge_statistics['ConditionFlow']['bleu_num']

    # recall
    refer_nodes = []  # record the nodes parsed by the refer to avoid calculating the same node repeatedly
    for refer_span_item in refer_spans:
        matched, refer_nodes, node_parse_num, node_match_num, edge_parse_num, edge_match_num, r_node_statistics, r_edge_statistics, r_agent_statistics, r_bleu_node_statistics, r_bleu_agent_statistics, r_bleu_edge_statistics = single_span_match(refer_span_item, all_parsed_spans, refer_nodes, spacy_tool, args)
        if matched:
            node_recall_total += node_parse_num
            node_recall_correct += node_match_num
            edge_recall_total += edge_parse_num
            edge_recall_correct += edge_match_num

            for node_type in node_types:
                node_statistics[node_type]['r_total'] += r_node_statistics[node_type]['total']
                node_statistics[node_type]['r_correct'] += r_node_statistics[node_type]['correct']
            for edge_type in edge_types:
                edge_statistics[edge_type]['r_total'] += r_edge_statistics[edge_type]['total']
                edge_statistics[edge_type]['r_correct'] += r_edge_statistics[edge_type]['correct']

            for agent_type in agent_types:
                agent_statistics[agent_type]['r_total'] += r_agent_statistics[agent_type]['total']
                agent_statistics[agent_type]['r_correct'] += r_agent_statistics[agent_type]['correct']

            # BLEU
            for node_type in bleu_node_types:
                bleu_node_statistics[node_type]['bleu_recall_total'] += r_bleu_node_statistics[node_type]['bleu_total']
                bleu_node_statistics[node_type]['bleu_recall_num'] += r_bleu_node_statistics[node_type]['bleu_num']
            for agent_type in bleu_agent_types:
                bleu_agent_statistics[agent_type]['bleu_recall_total'] += r_bleu_agent_statistics[agent_type]['bleu_total']
                bleu_agent_statistics[agent_type]['bleu_recall_num'] += r_bleu_agent_statistics[agent_type]['bleu_num']
            for edge_type in bleu_edge_types:
                bleu_edge_statistics[edge_type]['r_total'] += r_bleu_edge_statistics[edge_type]['total']
                bleu_edge_statistics[edge_type]['r_correct'] += r_bleu_edge_statistics[edge_type]['correct']
            # ConditionFlow
            bleu_edge_statistics['ConditionFlow']['bleu_recall_total'] += r_bleu_edge_statistics['ConditionFlow']['bleu_total']
            bleu_edge_statistics['ConditionFlow']['bleu_recall_num'] += r_bleu_edge_statistics['ConditionFlow']['bleu_num']

    return [node_precision_total, node_precision_correct, node_recall_total, node_recall_correct], \
                [edge_precision_total, edge_precision_correct, edge_recall_total, edge_recall_correct], node_statistics, edge_statistics, agent_statistics, bleu_node_statistics, bleu_agent_statistics, bleu_edge_statistics


class SpacyTool(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner", "morphologizer"])

    def get_lemma(self, text):
        doc = self.nlp(text)
        # [(X.text, X.lemma_) for X in doc]
        return [X.lemma_ for X in doc]

    def compute_bleu(self, pred, refer):
        if len(pred) == 0 or len(refer) == 0:
            return 0

        pred_lemma = self.get_lemma(pred)
        refer_lemma = self.get_lemma(refer)

        stop_words = [',', '.', '?', ';', ':', '!', '(', ')', '[', ']', '{', '}', '\'', '\"', '-', '_', 'a', 'the', 'an']
        pred_lemma = [word for word in pred_lemma if word not in stop_words]
        refer_lemma = [word for word in refer_lemma if word not in stop_words]

        if len(pred_lemma) == 0 or len(refer_lemma) == 0:
            return 0

        bleu_score = sentence_bleu([refer_lemma], pred_lemma, weights=(1, 0, 0, 0))
        bleu_score = round(bleu_score, 4)

        assert 0 <= bleu_score <= 1

        return bleu_score


class LexicalBased(object):
    def __init__(self, split_tag=None):
        self.metric_name = 'LexicalBased'
        self.args = {
            'span_threshold': 0.5,
            'node_match_threshold': 1,
            'same_parsed_match_threshold': 0.8,  # avoid same parsed match threshold
            'condition_match_threshold': 1,
            'agent_match_threshold': 1,
        }

        if split_tag:
            self.args['split_tag'] = split_tag
        else:
            self.args['split_tag'] = "\n"

        self.precision = {
            'nodes_total': 0,
            'nodes_correct': 0,

            'edges_total': 0,
            'edges_correct': 0,
        }
        self.recall = {
            'nodes_total': 0,
            'nodes_correct': 0,

            'edges_total': 0,
            'edges_correct': 0,
        }

        # node
        self.node_types = ['Activity', 'StartNode', 'EndNode', 'XOR', 'OR', 'AND', 'DataObject', 'TextAnnotation']
        self.node_statistics = {}
        for node_type in self.node_types:
            self.node_statistics[node_type] = {
                'p_total': 0,
                'p_correct': 0,
                'r_total': 0,
                'r_correct': 0,
            }

        # edge
        self.edge_types = ['SequenceFlow', 'Association']
        self.edge_statistics = {}
        for edge_type in self.edge_types:
            self.edge_statistics[edge_type] = {
                'p_total': 0,
                'p_correct': 0,
                'r_total': 0,
                'r_correct': 0,
            }

        # agent
        self.agent_types = ['Activity', 'StartNode', 'EndNode']
        self.agent_statistics = {}
        for agent_type in self.agent_types:
            self.agent_statistics[agent_type] = {
                'p_total': 0,
                'p_correct': 0,
                'r_total': 0,
                'r_correct': 0,
            }

        # BLEU based -----
        # node
        self.bleu_node_types = ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
        self.bleu_node_statistics = {}
        for node_type in self.bleu_node_types:
            self.bleu_node_statistics[node_type] = {
                'bleu_precision_total': 0,
                'bleu_precision_num': 0,
                'bleu_recall_total': 0,
                'bleu_recall_num': 0,
            }

        # agent
        self.bleu_agent_types = ['Activity', 'StartNode', 'EndNode']
        self.bleu_agent_statistics = {}
        for agent_type in self.bleu_agent_types:
            self.bleu_agent_statistics[agent_type] = {
                'bleu_precision_total': 0,
                'bleu_precision_num': 0,
                'bleu_recall_total': 0,
                'bleu_recall_num': 0,
            }

        # edge
        self.bleu_edge_types = ['SequenceFlow', 'Association']
        self.bleu_edge_statistics = {}
        for edge_type in self.bleu_edge_types:
            self.bleu_edge_statistics[edge_type] = {
                'p_total': 0,
                'p_correct': 0,
                'r_total': 0,
                'r_correct': 0,
            }

        # condition flow
        self.bleu_edge_statistics['ConditionFlow'] = {
            'bleu_precision_total': 0,
            'bleu_precision_num': 0,
            'bleu_recall_total': 0,
            'bleu_recall_num': 0,
        }

        self.spacy_tool = SpacyTool()

    def print_num(self):
        print('self.precision: ', self.precision)
        print('self.recall: ', self.recall)
        # node
        for node_type in self.node_types:
            print('node_type: ', node_type)
            print(self.node_statistics[node_type])
            print('-------\n')
        # edge
        for edge_type in self.edge_types:
            print('edge_type: ', edge_type)
            print(self.edge_statistics[edge_type])
            print('-------\n')

    def clear(self):
        for key in self.precision.keys():
            self.precision[key] = 0
        for key in self.recall.keys():
            self.recall[key] = 0

        for node_type in self.node_types:
            self.node_statistics[node_type]['p_total'] = 0
            self.node_statistics[node_type]['p_correct'] = 0
            self.node_statistics[node_type]['r_total'] = 0
            self.node_statistics[node_type]['r_correct'] = 0
        for edge_type in self.edge_types:
            self.edge_statistics[edge_type]['p_total'] = 0
            self.edge_statistics[edge_type]['p_correct'] = 0
            self.edge_statistics[edge_type]['r_total'] = 0
            self.edge_statistics[edge_type]['r_correct'] = 0

        for agent_type in self.agent_types:
            self.agent_statistics[agent_type]['p_total'] = 0
            self.agent_statistics[agent_type]['p_correct'] = 0
            self.agent_statistics[agent_type]['r_total'] = 0
            self.agent_statistics[agent_type]['r_correct'] = 0

        # BLEU based -----
        for node_type in self.bleu_node_types:
            self.bleu_node_statistics[node_type]['bleu_precision_total'] = 0
            self.bleu_node_statistics[node_type]['bleu_precision_num'] = 0
            self.bleu_node_statistics[node_type]['bleu_recall_total'] = 0
            self.bleu_node_statistics[node_type]['bleu_recall_num'] = 0

        for agent_type in self.bleu_agent_types:
            self.bleu_agent_statistics[agent_type]['bleu_precision_total'] = 0
            self.bleu_agent_statistics[agent_type]['bleu_precision_num'] = 0
            self.bleu_agent_statistics[agent_type]['bleu_recall_total'] = 0
            self.bleu_agent_statistics[agent_type]['bleu_recall_num'] = 0

        for edge_type in self.bleu_edge_types:
            self.bleu_edge_statistics[edge_type]['p_total'] = 0
            self.bleu_edge_statistics[edge_type]['p_correct'] = 0
            self.bleu_edge_statistics[edge_type]['r_total'] = 0
            self.bleu_edge_statistics[edge_type]['r_correct'] = 0

        self.bleu_edge_statistics['ConditionFlow']['bleu_precision_total'] = 0
        self.bleu_edge_statistics['ConditionFlow']['bleu_precision_num'] = 0
        self.bleu_edge_statistics['ConditionFlow']['bleu_recall_total'] = 0
        self.bleu_edge_statistics['ConditionFlow']['bleu_recall_num'] = 0

    def update(self, model_output, all_triples):
        # 1. parse spans
        all_parsed_spans = parse_model_output(model_output, self.args['split_tag'])

        # 2. match
        nodes_result, edges_result, node_statistics, edge_statistics, agent_statistics, bleu_node_statistics, bleu_agent_statistics, bleu_edge_statistics = spans_exact_match(all_parsed_spans, all_triples, self.spacy_tool, self.args)
        self.precision['nodes_total'] += nodes_result[0]
        self.precision['nodes_correct'] += nodes_result[1]
        self.recall['nodes_total'] += nodes_result[2]
        self.recall['nodes_correct'] += nodes_result[3]

        self.precision['edges_total'] += edges_result[0]
        self.precision['edges_correct'] += edges_result[1]
        self.recall['edges_total'] += edges_result[2]
        self.recall['edges_correct'] += edges_result[3]

        # statistics
        for node_type in self.node_types:
            self.node_statistics[node_type]['p_total'] += node_statistics[node_type]['p_total']
            self.node_statistics[node_type]['p_correct'] += node_statistics[node_type]['p_correct']
            self.node_statistics[node_type]['r_total'] += node_statistics[node_type]['r_total']
            self.node_statistics[node_type]['r_correct'] += node_statistics[node_type]['r_correct']
        for edge_type in self.edge_types:
            self.edge_statistics[edge_type]['p_total'] += edge_statistics[edge_type]['p_total']
            self.edge_statistics[edge_type]['p_correct'] += edge_statistics[edge_type]['p_correct']
            self.edge_statistics[edge_type]['r_total'] += edge_statistics[edge_type]['r_total']
            self.edge_statistics[edge_type]['r_correct'] += edge_statistics[edge_type]['r_correct']

        # agent
        for agent_type in self.agent_types:
            self.agent_statistics[agent_type]['p_total'] += agent_statistics[agent_type]['p_total']
            self.agent_statistics[agent_type]['p_correct'] += agent_statistics[agent_type]['p_correct']
            self.agent_statistics[agent_type]['r_total'] += agent_statistics[agent_type]['r_total']
            self.agent_statistics[agent_type]['r_correct'] += agent_statistics[agent_type]['r_correct']

        # BLEU
        for node_type in self.bleu_node_types:
            self.bleu_node_statistics[node_type]['bleu_precision_total'] += bleu_node_statistics[node_type]['bleu_precision_total']
            self.bleu_node_statistics[node_type]['bleu_precision_num'] += bleu_node_statistics[node_type]['bleu_precision_num']
            self.bleu_node_statistics[node_type]['bleu_recall_total'] += bleu_node_statistics[node_type]['bleu_recall_total']
            self.bleu_node_statistics[node_type]['bleu_recall_num'] += bleu_node_statistics[node_type]['bleu_recall_num']
        for agent_type in self.bleu_agent_types:
            self.bleu_agent_statistics[agent_type]['bleu_precision_total'] += bleu_agent_statistics[agent_type]['bleu_precision_total']
            self.bleu_agent_statistics[agent_type]['bleu_precision_num'] += bleu_agent_statistics[agent_type]['bleu_precision_num']
            self.bleu_agent_statistics[agent_type]['bleu_recall_total'] += bleu_agent_statistics[agent_type]['bleu_recall_total']
            self.bleu_agent_statistics[agent_type]['bleu_recall_num'] += bleu_agent_statistics[agent_type]['bleu_recall_num']
        for edge_type in self.bleu_edge_types:
            self.bleu_edge_statistics[edge_type]['p_total'] += bleu_edge_statistics[edge_type]['p_total']
            self.bleu_edge_statistics[edge_type]['p_correct'] += bleu_edge_statistics[edge_type]['p_correct']
            self.bleu_edge_statistics[edge_type]['r_total'] += bleu_edge_statistics[edge_type]['r_total']
            self.bleu_edge_statistics[edge_type]['r_correct'] += bleu_edge_statistics[edge_type]['r_correct']

        # ConditionFlow
        self.bleu_edge_statistics['ConditionFlow']['bleu_precision_total'] += bleu_edge_statistics['ConditionFlow']['bleu_precision_total']
        self.bleu_edge_statistics['ConditionFlow']['bleu_precision_num'] += bleu_edge_statistics['ConditionFlow']['bleu_precision_num']
        self.bleu_edge_statistics['ConditionFlow']['bleu_recall_total'] += bleu_edge_statistics['ConditionFlow']['bleu_recall_total']
        self.bleu_edge_statistics['ConditionFlow']['bleu_recall_num'] += bleu_edge_statistics['ConditionFlow']['bleu_recall_num']

        # compute scores
        cur_node_p = (nodes_result[1] / nodes_result[0]) if nodes_result[0] > 0 else 0
        cur_node_r = (nodes_result[3] / nodes_result[2]) if nodes_result[2] > 0 else 0
        cur_node_f1 = (2 * cur_node_p * cur_node_r) / (cur_node_p + cur_node_r) if (cur_node_p + cur_node_r) > 0 else 0

        cur_edge_p = (edges_result[1] / edges_result[0]) if edges_result[0] > 0 else 0
        cur_edge_r = (edges_result[3] / edges_result[2]) if edges_result[2] > 0 else 0
        cur_edge_f1 = (2 * cur_edge_p * cur_edge_r) / (cur_edge_p + cur_edge_r) if (cur_edge_p + cur_edge_r) > 0 else 0

        sample_result = {
            'node_precision': round(cur_node_p, 4),
            'node_recall': round(cur_node_r, 4),
            'node_f1': round(cur_node_f1, 4),
            'edge_precision': round(cur_edge_p, 4),
            'edge_recall': round(cur_edge_r, 4),
            'edge_f1': round(cur_edge_f1, 4),
        }

        # for different types
        group_eva_results = {}
        group_eva_map_node = {
            'Action': ['Activity', 'StartNode', 'EndNode'],
            'Gateway': ['XOR', 'OR', 'AND'],
            'Constraint': ['DataObject', 'TextAnnotation'],
        }
        group_eva_map_edge = {
            'SequentialEdge': ['SequenceFlow'],
            'ConstraintEdge': ['Association'],
        }
        group_eva_map_agent = {
            'Actor': ['Activity', 'StartNode', 'EndNode'],
        }

        # group_eva_map_node
        for group_type in group_eva_map_node.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for node_type in group_eva_map_node[group_type]:
                p_total += node_statistics[node_type]['p_total']
                p_correct += node_statistics[node_type]['p_correct']
                r_total += node_statistics[node_type]['r_total']
                r_correct += node_statistics[node_type]['r_correct']
            n_precision = (p_correct / p_total) if p_total > 0 else 0
            n_recall = (r_correct / r_total) if r_total > 0 else 0
            n_f1 = (2 * n_precision * n_recall) / (n_precision + n_recall) if (n_precision + n_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(n_precision, 4),
                'recall': round(n_recall, 4),
                'f1': round(n_f1, 4),
            }

        # group_eva_map_edge
        for group_type in group_eva_map_edge.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for edge_type in group_eva_map_edge[group_type]:
                p_total += edge_statistics[edge_type]['p_total']
                p_correct += edge_statistics[edge_type]['p_correct']
                r_total += edge_statistics[edge_type]['r_total']
                r_correct += edge_statistics[edge_type]['r_correct']
            e_precision = (p_correct / p_total) if p_total > 0 else 0
            e_recall = (r_correct / r_total) if r_total > 0 else 0
            e_f1 = (2 * e_precision * e_recall) / (e_precision + e_recall) if (e_precision + e_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(e_precision, 4),
                'recall': round(e_recall, 4),
                'f1': round(e_f1, 4),
            }

        # group_eva_map_agent
        for group_type in group_eva_map_agent.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for agent_type in group_eva_map_agent[group_type]:
                p_total += agent_statistics[agent_type]['p_total']
                p_correct += agent_statistics[agent_type]['p_correct']
                r_total += agent_statistics[agent_type]['r_total']
                r_correct += agent_statistics[agent_type]['r_correct']
            a_precision = (p_correct / p_total) if p_total > 0 else 0
            a_recall = (r_correct / r_total) if r_total > 0 else 0
            a_f1 = (2 * a_precision * a_recall) / (a_precision + a_recall) if (a_precision + a_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(a_precision, 4),
                'recall': round(a_recall, 4),
                'f1': round(a_f1, 4),
            }

        sample_result['group_eva_results'] = group_eva_results

        return sample_result

    def cur_results(self):
        nodes_precision = (self.precision['nodes_correct'] / self.precision['nodes_total']) if self.precision['nodes_total'] > 0 else 0
        nodes_recall = (self.recall['nodes_correct'] / self.recall['nodes_total']) if self.recall['nodes_total'] > 0 else 0
        nodes_f1 = (2 * nodes_precision * nodes_recall) / (nodes_precision + nodes_recall) if (nodes_precision + nodes_recall) > 0 else 0
        nodes_f1 = round(nodes_f1, 4)
        edges_precision = (self.precision['edges_correct'] / self.precision['edges_total']) if self.precision['edges_total'] > 0 else 0
        edges_recall = (self.recall['edges_correct'] / self.recall['edges_total']) if self.recall['edges_total'] > 0 else 0
        edges_f1 = (2 * edges_precision * edges_recall) / (edges_precision + edges_recall) if (edges_precision + edges_recall) > 0 else 0
        edges_f1 = round(edges_f1, 4)
        results = {
            'nodes_f1': nodes_f1,
            'edges_f1': edges_f1,
        }
        return results

    def get_results(self):
        # compute precision
        nodes_precision = (self.precision['nodes_correct'] / self.precision['nodes_total']) if self.precision['nodes_total'] > 0 else 0
        nodes_recall = (self.recall['nodes_correct'] / self.recall['nodes_total']) if self.recall['nodes_total'] > 0 else 0
        nodes_f1 = (2 * nodes_precision * nodes_recall) / (nodes_precision + nodes_recall) if (nodes_precision + nodes_recall) > 0 else 0
        nodes_precision = round(nodes_precision, 4)
        nodes_recall = round(nodes_recall, 4)
        nodes_f1 = round(nodes_f1, 4)

        edges_precision = (self.precision['edges_correct'] / self.precision['edges_total']) if self.precision['edges_total'] > 0 else 0
        edges_recall = (self.recall['edges_correct'] / self.recall['edges_total']) if self.recall['edges_total'] > 0 else 0
        edges_f1 = (2 * edges_precision * edges_recall) / (edges_precision + edges_recall) if (edges_precision + edges_recall) > 0 else 0
        edges_precision = round(edges_precision, 4)
        edges_recall = round(edges_recall, 4)
        edges_f1 = round(edges_f1, 4)

        results = {
            'nodes_precision': nodes_precision,
            'nodes_recall': nodes_recall,
            'nodes_f1': nodes_f1,
            'edges_precision': edges_precision,
            'edges_recall': edges_recall,
            'edges_f1': edges_f1,
        }
        return results

    def get_detailed_results(self):
        # compute scores for each node type
        node_detailed_results = {}
        for node_type in self.node_types:
            precision = (self.node_statistics[node_type]['p_correct'] / self.node_statistics[node_type]['p_total']) if self.node_statistics[node_type]['p_total'] > 0 else 0
            recall = (self.node_statistics[node_type]['r_correct'] / self.node_statistics[node_type]['r_total']) if self.node_statistics[node_type]['r_total'] > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precision = round(precision, 4)
            recall = round(recall, 4)
            f1 = round(f1, 4)
            node_detailed_results[node_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        # compute scores for each edge type
        edge_detailed_results = {}
        for edge_type in self.edge_types:
            precision = (self.edge_statistics[edge_type]['p_correct'] / self.edge_statistics[edge_type]['p_total']) if self.edge_statistics[edge_type]['p_total'] > 0 else 0
            recall = (self.edge_statistics[edge_type]['r_correct'] / self.edge_statistics[edge_type]['r_total']) if self.edge_statistics[edge_type]['r_total'] > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precision = round(precision, 4)
            recall = round(recall, 4)
            f1 = round(f1, 4)
            edge_detailed_results[edge_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

        # compute scores for all nodes
        node_total_results = {}
        p_total = 0
        p_correct = 0
        r_total = 0
        r_correct = 0
        for node_type in self.node_types:
            p_total += self.node_statistics[node_type]['p_total']
            p_correct += self.node_statistics[node_type]['p_correct']
            r_total += self.node_statistics[node_type]['r_total']
            r_correct += self.node_statistics[node_type]['r_correct']
        n_precision = (p_correct / p_total) if p_total > 0 else 0
        n_recall = (r_correct / r_total) if r_total > 0 else 0
        n_f1 = (2 * n_precision * n_recall) / (n_precision + n_recall) if (n_precision + n_recall) > 0 else 0
        node_total_results['precision'] = round(n_precision, 4)
        node_total_results['recall'] = round(n_recall, 4)
        node_total_results['f1'] = round(n_f1, 4)

        # compute scores for all edges
        edge_total_results = {}
        p_total = 0
        p_correct = 0
        r_total = 0
        r_correct = 0
        for edge_type in self.edge_types:
            p_total += self.edge_statistics[edge_type]['p_total']
            p_correct += self.edge_statistics[edge_type]['p_correct']
            r_total += self.edge_statistics[edge_type]['r_total']
            r_correct += self.edge_statistics[edge_type]['r_correct']
        e_precision = (p_correct / p_total) if p_total > 0 else 0
        e_recall = (r_correct / r_total) if r_total > 0 else 0
        e_f1 = (2 * e_precision * e_recall) / (e_precision + e_recall) if (e_precision + e_recall) > 0 else 0
        edge_total_results['precision'] = round(e_precision, 4)
        edge_total_results['recall'] = round(e_recall, 4)
        edge_total_results['f1'] = round(e_f1, 4)

        # node_eva_results ---------------------------------------------------------------------------------
        node_eva_results = {}
        node_eva_map = {
            'Activity': ['Activity', 'StartNode', 'EndNode'],
            'XOR': ['XOR'],
            'OR': ['OR'],
            'AND': ['AND'],
            'DataObject': ['DataObject'],
            'TextAnnotation': ['TextAnnotation'],
        }

        for nodes_type in node_eva_map.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for node_type in node_eva_map[nodes_type]:
                p_total += self.node_statistics[node_type]['p_total']
                p_correct += self.node_statistics[node_type]['p_correct']
                r_total += self.node_statistics[node_type]['r_total']
                r_correct += self.node_statistics[node_type]['r_correct']
            n_precision = (p_correct / p_total) if p_total > 0 else 0
            n_recall = (r_correct / r_total) if r_total > 0 else 0
            n_f1 = (2 * n_precision * n_recall) / (n_precision + n_recall) if (n_precision + n_recall) > 0 else 0
            node_eva_results[nodes_type] = {
                'precision': round(n_precision, 4),
                'recall': round(n_recall, 4),
                'f1': round(n_f1, 4),
            }

        # others ---------------------------------------------------------------------------------
        other_results = {}

        # agent ---------------------------------------------------------------------------------
        agent_results = {}
        agent_p_total = sum([self.agent_statistics[agent_type]['p_total'] for agent_type in self.agent_types])
        agent_p_correct = sum([self.agent_statistics[agent_type]['p_correct'] for agent_type in self.agent_types])
        agent_r_total = sum([self.agent_statistics[agent_type]['r_total'] for agent_type in self.agent_types])
        agent_r_correct = sum([self.agent_statistics[agent_type]['r_correct'] for agent_type in self.agent_types])
        agent_precision = (agent_p_correct / agent_p_total) if agent_p_total > 0 else 0
        agent_recall = (agent_r_correct / agent_r_total) if agent_r_total > 0 else 0
        agent_f1 = (2 * agent_precision * agent_recall) / (agent_precision + agent_recall) if (agent_precision + agent_recall) > 0 else 0
        agent_results['total'] = {
            'precision': round(agent_precision, 4),
            'recall': round(agent_recall, 4),
            'f1': round(agent_f1, 4),
        }

        for agent_type in self.agent_types:
            p_total = self.agent_statistics[agent_type]['p_total']
            p_correct = self.agent_statistics[agent_type]['p_correct']
            r_total = self.agent_statistics[agent_type]['r_total']
            r_correct = self.agent_statistics[agent_type]['r_correct']
            a_precision = (p_correct / p_total) if p_total > 0 else 0
            a_recall = (r_correct / r_total) if r_total > 0 else 0
            a_f1 = (2 * a_precision * a_recall) / (a_precision + a_recall) if (a_precision + a_recall) > 0 else 0
            agent_results[agent_type] = {
                'precision': round(a_precision, 4),
                'recall': round(a_recall, 4),
                'f1': round(a_f1, 4),
            }
        other_results['agent'] = agent_results

        # groups -----------------------------------------------------------------------------------------------
        group_eva_results = {}
        group_eva_map_node = {
            'Action': ['Activity', 'StartNode', 'EndNode'],
            'Gateway': ['XOR', 'OR', 'AND'],
            'Constraint': ['DataObject', 'TextAnnotation'],
        }
        group_eva_map_edge = {
            'SequentialEdge': ['SequenceFlow'],
            'ConstraintEdge': ['Association'],
        }
        group_eva_map_agent = {
            'Actor': ['Activity', 'StartNode', 'EndNode'],
        }
        # group_eva_map_node
        for group_type in group_eva_map_node.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for node_type in group_eva_map_node[group_type]:
                p_total += self.node_statistics[node_type]['p_total']
                p_correct += self.node_statistics[node_type]['p_correct']
                r_total += self.node_statistics[node_type]['r_total']
                r_correct += self.node_statistics[node_type]['r_correct']
            n_precision = (p_correct / p_total) if p_total > 0 else 0
            n_recall = (r_correct / r_total) if r_total > 0 else 0
            n_f1 = (2 * n_precision * n_recall) / (n_precision + n_recall) if (n_precision + n_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(n_precision, 4),
                'recall': round(n_recall, 4),
                'f1': round(n_f1, 4),
            }
        # group_eva_map_edge
        for group_type in group_eva_map_edge.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for edge_type in group_eva_map_edge[group_type]:
                p_total += self.edge_statistics[edge_type]['p_total']
                p_correct += self.edge_statistics[edge_type]['p_correct']
                r_total += self.edge_statistics[edge_type]['r_total']
                r_correct += self.edge_statistics[edge_type]['r_correct']
            e_precision = (p_correct / p_total) if p_total > 0 else 0
            e_recall = (r_correct / r_total) if r_total > 0 else 0
            e_f1 = (2 * e_precision * e_recall) / (e_precision + e_recall) if (e_precision + e_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(e_precision, 4),
                'recall': round(e_recall, 4),
                'f1': round(e_f1, 4),
            }
        # group_eva_map_agent
        for group_type in group_eva_map_agent.keys():
            p_total = 0
            p_correct = 0
            r_total = 0
            r_correct = 0
            for agent_type in group_eva_map_agent[group_type]:
                p_total += self.agent_statistics[agent_type]['p_total']
                p_correct += self.agent_statistics[agent_type]['p_correct']
                r_total += self.agent_statistics[agent_type]['r_total']
                r_correct += self.agent_statistics[agent_type]['r_correct']
            a_precision = (p_correct / p_total) if p_total > 0 else 0
            a_recall = (r_correct / r_total) if r_total > 0 else 0
            a_f1 = (2 * a_precision * a_recall) / (a_precision + a_recall) if (a_precision + a_recall) > 0 else 0
            group_eva_results[group_type] = {
                'precision': round(a_precision, 4),
                'recall': round(a_recall, 4),
                'f1': round(a_f1, 4),
            }

        other_results['group'] = group_eva_results

        # BLEU
        node_bleu_results = {}

        # ['Activity', 'StartNode', 'EndNode', 'DataObject', 'TextAnnotation']
        group_map_bleu_node = {
            'Action': ['Activity', 'StartNode', 'EndNode'],
            'DataObject': ['DataObject'],
            'TextAnnotation': ['TextAnnotation'],
            'Constraint': ['DataObject', 'TextAnnotation'],
        }

        for key in group_map_bleu_node.keys():
            bleu_precision_total = 0
            bleu_precision_num = 0
            bleu_recall_total = 0
            bleu_recall_num = 0
            for node_type in group_map_bleu_node[key]:
                bleu_precision_total += self.bleu_node_statistics[node_type]['bleu_precision_total']
                bleu_precision_num += self.bleu_node_statistics[node_type]['bleu_precision_num']
                bleu_recall_total += self.bleu_node_statistics[node_type]['bleu_recall_total']
                bleu_recall_num += self.bleu_node_statistics[node_type]['bleu_recall_num']
            bleu_precision = (bleu_precision_total / bleu_precision_num) if bleu_precision_num > 0 else 0
            bleu_recall = (bleu_recall_total / bleu_recall_num) if bleu_recall_num > 0 else 0
            bleu_f1 = (2 * bleu_precision * bleu_recall) / (bleu_precision + bleu_recall) if (bleu_precision + bleu_recall) > 0 else 0
            node_bleu_results[key] = {
                'precision': round(bleu_precision, 4),
                'recall': round(bleu_recall, 4),
                'f1': round(bleu_f1, 4),
            }

        # agent
        agent_bleu_results = {}
        group_map_bleu_agent = {
            'Actor': ['Activity', 'StartNode', 'EndNode'],
        }
        for key in group_map_bleu_agent.keys():
            bleu_precision_total = 0
            bleu_precision_num = 0
            bleu_recall_total = 0
            bleu_recall_num = 0
            for agent_type in group_map_bleu_agent[key]:
                bleu_precision_total += self.bleu_agent_statistics[agent_type]['bleu_precision_total']
                bleu_precision_num += self.bleu_agent_statistics[agent_type]['bleu_precision_num']
                bleu_recall_total += self.bleu_agent_statistics[agent_type]['bleu_recall_total']
                bleu_recall_num += self.bleu_agent_statistics[agent_type]['bleu_recall_num']
            bleu_precision = (bleu_precision_total / bleu_precision_num) if bleu_precision_num > 0 else 0
            bleu_recall = (bleu_recall_total / bleu_recall_num) if bleu_recall_num > 0 else 0
            bleu_f1 = (2 * bleu_precision * bleu_recall) / (bleu_precision + bleu_recall) if (bleu_precision + bleu_recall) > 0 else 0
            agent_bleu_results[key] = {
                'precision': round(bleu_precision, 4),
                'recall': round(bleu_recall, 4),
                'f1': round(bleu_f1, 4),
            }

        # edge
        bleu_edge_types = ['SequenceFlow', 'Association']
        edge_bleu_results = {}
        for edge_type in bleu_edge_types:
            p_total = self.bleu_edge_statistics[edge_type]['p_total']
            p_correct = self.bleu_edge_statistics[edge_type]['p_correct']
            r_total = self.bleu_edge_statistics[edge_type]['r_total']
            r_correct = self.bleu_edge_statistics[edge_type]['r_correct']
            e_precision = (p_correct / p_total) if p_total > 0 else 0
            e_recall = (r_correct / r_total) if r_total > 0 else 0
            e_f1 = (2 * e_precision * e_recall) / (e_precision + e_recall) if (e_precision + e_recall) > 0 else 0
            edge_bleu_results[edge_type] = {
                'precision': round(e_precision, 4),
                'recall': round(e_recall, 4),
                'f1': round(e_f1, 4),
            }
        # ConditionFlow
        bleu_precision_total = self.bleu_edge_statistics['ConditionFlow']['bleu_precision_total']
        bleu_precision_num = self.bleu_edge_statistics['ConditionFlow']['bleu_precision_num']
        bleu_recall_total = self.bleu_edge_statistics['ConditionFlow']['bleu_recall_total']
        bleu_recall_num = self.bleu_edge_statistics['ConditionFlow']['bleu_recall_num']
        bleu_precision = (bleu_precision_total / bleu_precision_num) if bleu_precision_num > 0 else 0
        bleu_recall = (bleu_recall_total / bleu_recall_num) if bleu_recall_num > 0 else 0
        bleu_f1 = (2 * bleu_precision * bleu_recall) / (bleu_precision + bleu_recall) if (bleu_precision + bleu_recall) > 0 else 0
        edge_bleu_results['ConditionFlow'] = {
            'precision': round(bleu_precision, 4),
            'recall': round(bleu_recall, 4),
            'f1': round(bleu_f1, 4),
        }

        other_results['bleu_node'] = node_bleu_results
        other_results['bleu_agent'] = agent_bleu_results
        other_results['bleu_edge'] = edge_bleu_results

        return node_detailed_results, edge_detailed_results, node_total_results, edge_total_results, node_eva_results, other_results
