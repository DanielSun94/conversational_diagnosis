import csv
import pickle
from itertools import islice
import os
import random
from transformers import AutoTokenizer, AutoModel
import torch
import json
from screen_agent.screen_config import (fixed_question_answer_folder, question_embedding_folder)


device = 'cuda:0'


def read_id_order_seq(path, random_seed=715):
    id_list = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            unified_id, _, __, note_id = line[:4]
            key = unified_id + '-' + note_id
            id_list.append(key)
    random.Random(random_seed).shuffle(id_list)
    return id_list


def get_next_batch_id(data_dict, transformed_id_dict, id_dict, batch_size):
    id_list = []
    for key in id_dict:
        if key in data_dict and key not in transformed_id_dict:
            id_list.append(key)
        if len(id_list) >= batch_size:
            break
    return id_list

def read_data(general_folder):
    data_dict = dict()
    folder_list = os.listdir(general_folder)
    for folder in folder_list:
        folder_path = os.path.join(general_folder, folder)
        file_list = os.listdir(folder_path)
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
            file_key = file[:-5]
            data_dict[file_key] = data
    return data_dict


def reorganize_data(data_dict):
    data_list = []
    for key in data_dict:
        # 只放既往史是因为数据里只有既往史是可用的
        try:
            en_data = data_dict[key]['en']
            en_text = 'Past Medical History: {}.'.format(en_data['past_medical_history']
            )
            zh_data = data_dict[key]['zh']
            # zh_text = '基本信息：{}.\n既往史：{}.\n家族史：{}.\n'.format(
            #     zh_data['基本信息'], zh_data['既往史'], zh_data['家族史']
            # )
            zh_text = '既往史：{}。'.format(zh_data['既往史'])
            data_list.append([key, en_text, zh_text])
        except Exception as e:
            print('key error: {}, {}'.format(key, data_dict[key]))
        # if len(data_list) > 100:
        #     break
    return data_list


def generate_embedding(data_list):
    chinese_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    chinese_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').to(device)
    chinese_model.eval()
    english_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    english_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(device)
    english_model.eval()
    zh_embedding_dict, en_embedding_dict = dict(), dict()
    for i, (key, en_text, zh_text) in enumerate(data_list):
        english_encoded_input = english_tokenizer(en_text, max_length=512,
                                                  padding=True, truncation=True, return_tensors='pt')
        english_encoded_input = english_encoded_input.to(device)
        with torch.no_grad():
            english_model_output = english_model(**english_encoded_input)
            english_embedding = english_model_output[0][:, 0]
        english_embedding = torch.nn.functional.normalize(english_embedding, p=2, dim=1)
        english_embedding = [item for item in english_embedding.detach().to('cpu').numpy()[0]]

        chinese_encoded_input = chinese_tokenizer(en_text, max_length=512,
                                                  padding=True, truncation=True, return_tensors='pt')
        chinese_encoded_input = chinese_encoded_input.to(device)
        with torch.no_grad():
            chinese_model_output = chinese_model(**chinese_encoded_input)
            chinese_embedding = chinese_model_output[0][:, 0]
        chinese_embedding = torch.nn.functional.normalize(chinese_embedding, p=2, dim=1)
        chinese_embedding = [item for item in chinese_embedding.detach().to('cpu').numpy()[0]]

        folder_index = i // 400
        os.makedirs(os.path.join(question_embedding_folder, str(folder_index)), exist_ok=True)
        path = os.path.join(question_embedding_folder, str(folder_index), key + '.pkl')
        pickle.dump({'chinese': chinese_embedding, 'english': english_embedding}, open(path, 'wb'))
        print('success')
    return en_embedding_dict, zh_embedding_dict


def main():
    data_dict = read_data(fixed_question_answer_folder)
    data_list = reorganize_data(data_dict)
    en_embedding_dict, zh_embedding_dict = generate_embedding(data_list)


if __name__ == '__main__':
    main()
