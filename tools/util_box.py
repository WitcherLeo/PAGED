import json
import jsonlines
from typing import List
import pandas as pd
import csv


def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    print('read json file: %s' % file_path)
    return data


def write_json(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)
    print('write json file: %s' % file_path)


def read_jsonl(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = [json.loads(line) for line in f]
    print('read jsonl file: %s' % file_path)
    return data


def write_jsonl(file_path, data: List, encoding='utf-8'):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)
    print('write jsonl file: %s' % file_path)


def read_txt(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = f.read()
    print('read txt file: %s' % file_path)
    return data


def save_txt(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(data)
    print('write txt file: %s' % file_path)


def save_txt_append(file_path, data, encoding='utf-8'):
    with open(file_path, 'a', encoding=encoding) as f:
        f.write(data)
    print('write txt file: %s' % file_path)


def read_csv(file_path, encoding='utf-8'):
    data = pd.read_csv(file_path, encoding=encoding)
    print('read csv file: %s' % file_path)
    return data


def write_texts_list_to_tsv(file_path, texts_list, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as f:
        writer = csv.writer(f, delimiter='\t')
        for text in texts_list:
            writer.writerow([text])
    print('write tsv file: %s' % file_path)


def add2dict(dic, key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1
