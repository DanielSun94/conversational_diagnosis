import csv
import pickle
from screnning_agent.config import (reserved_note_path, embedding_transformed_id_list_file_path, history_folder,
                                    open_ai_embedding_folder)
from itertools import islice
import os
import threading
from datetime import datetime
import random
from transformers import AutoTokenizer, AutoModel
import torch
from llm_util import call_open_ai_embedding


THREAD_NUM = 15
MAX_NUM = 40000
FOLDER_SIZE = 400


def read_reserved_note(path):
    data_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            unified_id, note_id = line[0], line[3]
            history = line[10].replace('\n', '  ')
            data_dict[unified_id] = history
    return data_dict


def translate_history(data_dict):
    if os.path.exists(embedding_transformed_id_list_file_path):
        transformed_id_dict = dict()
        with open(embedding_transformed_id_list_file_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                transformed_id_dict[line[0]] = line[1]
    else:
        transformed_id_dict = dict()
    id_seq_list = list(data_dict.keys())
    random.Random(715).shuffle(id_seq_list)
    count_id = len(transformed_id_dict)
    while count_id < MAX_NUM:
        start_time = datetime.now()
        unified_ids = get_next_batch_id(data_dict, transformed_id_dict, id_seq_list, THREAD_NUM)
        if count_id >= MAX_NUM or len(unified_ids) == 0:
            break

        results = [[None, None, None, None, None]] * len(unified_ids)
        threads = []
        for i, unified_id in enumerate(unified_ids):
            thread = threading.Thread(
                target=generate_translation,
                args=(data_dict, unified_id, results, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for result in results:
            if result[0] is None or result[1] is None or result[2] is None or result[3] is None or result[4] is None:
                continue
            chinese_translation, english, _, __, unified_id = result
            formatted_num = "{:05d}".format(len(transformed_id_dict) // FOLDER_SIZE)
            folder_path = os.path.join(history_folder, formatted_num)
            os.makedirs(folder_path, exist_ok=True)

            obj_path = os.path.join(folder_path, unified_id + '.pkl')
            pickle.dump(
                {'chinese': chinese_translation, 'english': english},
                open(obj_path, 'wb')
            )

            transformed_id_dict[unified_id] = formatted_num
            print('unified_id: {} success'.format(unified_id))
            count_id += 1

        with open(embedding_transformed_id_list_file_path, 'w', encoding='utf-8-sig', newline='') as f:
            data_to_write = [['unified_id', 'folder_num']]
            for unified_id in transformed_id_dict:
                data_to_write.append([unified_id, transformed_id_dict[unified_id]])
            csv.writer(f).writerows(data_to_write)

        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Time difference: {time_diff}")


def generate_translation(data_dict, unified_id, results, index):
    content = data_dict[unified_id]
    init_prompt = "Please translate the following content into Chinese, please only response Chinese:\n\n"
    prompt = init_prompt + content
    chinese_translation = call_open_ai_embedding(prompt, model_name='gpt-3.5')
    results[index] = chinese_translation, content, '', '', unified_id
    print('unified_id: {}\n english: {}\n\n chinese: {}\n\n'.format(unified_id, content, chinese_translation))
    return


def get_save_embedding(data_dict, unified_id, success_id_set, id_dict):
    idx = id_dict[unified_id]
    folder = os.path.join(open_ai_embedding_folder, str(idx))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, unified_id + '.pkl')

    success_flag = False
    if os.path.exists(path):
        result = pickle.load(open(path, 'rb'))
        if 'chinese' in result and 'english' in result:
            if isinstance(result['chinese'], list) and isinstance(result['english'], list):
                success_flag = True
    if not success_flag:
        results = dict()
        english = data_dict[unified_id]['english']
        chinese = data_dict[unified_id]['chinese']
        chinese_embedding = call_open_ai_embedding(chinese)
        english_embedding = call_open_ai_embedding(english)
        results['chinese'] = chinese_embedding
        results['english'] = english_embedding
        pickle.dump(results, open(path, 'wb'))
    success_id_set.add(unified_id)
    print('unified_id: {} success'.format(unified_id))


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
            data = pickle.load(open(file_path, 'rb'))
            file_key = file.strip().split('.')[0]
            data_dict[file_key] = data
    return data_dict


def generate_embedding(data_dict, use_open_ai=True):
    if not use_open_ai:
        index = 0
        chinese_sentence_list = []
        english_sentence_list = []
        key_list = []
        embedding_dict = dict()
        chinese_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
        chinese_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').to('cuda:0')
        chinese_model.eval()
        english_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        english_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to('cuda:1')
        english_model.eval()

        for key in data_dict:
            key_list.append(key)
            english_sentence_list.append(data_dict[key]['english'])
            chinese_sentence_list.append(data_dict[key]['chinese'])
            if (index == len(data_dict) - 1 or index % 512 == 0) and index > 0:
                if len(key_list) == 0:
                    continue
                encoded_input = chinese_tokenizer(chinese_sentence_list, max_length=512,
                                                  padding=True, truncation=True, return_tensors='pt')
                encoded_input = encoded_input.to('cuda:0')
                with torch.no_grad():
                    model_output = chinese_model(**encoded_input)
                    chinese_embeddings = model_output[0][:, 0]
                chinese_embeddings = torch.nn.functional.normalize(chinese_embeddings, p=2, dim=1)
                encoded_input = english_tokenizer(english_sentence_list, max_length=512,
                                                  padding=True, truncation=True, return_tensors='pt')
                encoded_input = encoded_input.to('cuda:1')
                with torch.no_grad():
                    model_output = english_model(**encoded_input.to('cuda:1'))
                    english_embeddings = model_output[0][:, 0]
                english_embeddings = torch.nn.functional.normalize(english_embeddings, p=2, dim=1)
                for key_, chinese_embedding, english_embedding in \
                        zip(key_list, chinese_embeddings, english_embeddings):
                    embedding_dict[key_] = {'chinese': chinese_embedding.detach().cpu().numpy(),
                                            'english': english_embedding.detach().cpu().numpy()}
                chinese_sentence_list = []
                english_sentence_list = []
                key_list = []
                print('len: {}'.format(len(embedding_dict)))
            index += 1
    else:
        id_seq_list = list(data_dict.keys())
        random.Random(715).shuffle(id_seq_list)
        id_dict = dict()
        for i, unified_id in enumerate(id_seq_list):
            id_dict[unified_id] = i // FOLDER_SIZE

        count_id = 0
        success_id_set = set()
        while count_id < len(id_seq_list):
            start_time = datetime.now()
            unified_ids = get_next_batch_id(data_dict, success_id_set, id_dict, THREAD_NUM)
            if count_id >= MAX_NUM or len(unified_ids) == 0:
                break

            threads = []
            for i, unified_id in enumerate(unified_ids):
                thread = threading.Thread(
                    target=get_save_embedding,
                    args=(data_dict, unified_id, success_id_set, id_dict))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = datetime.now()
            time_diff = end_time - start_time
            print(f"Time difference: {time_diff}")
            print(f"current embedding dict length: {len(success_id_set)}")
    return None


# 2
def main():
    use_open_ai = True
    print('start')
    # local
    data_dict = read_reserved_note(reserved_note_path)
    translate_history(data_dict)
    print('success')
    data_dict = read_data(history_folder)
    generate_embedding(data_dict, use_open_ai=use_open_ai)
    # assert len(data_dict) == len(embedding_dict)



if __name__ == '__main__':
    main()
