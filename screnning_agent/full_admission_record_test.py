import csv
import math
from itertools import islice
from read_data import get_structured_diagnosis
from config import (reserved_note_path, open_ai_full_admission_embedding_folder)
from logger import logger
import random
from datetime import datetime
import threading
import numpy as np
import os
from llm_util import call_open_ai_embedding
import pickle
from sklearn.neural_network import MLPClassifier
from eval import top_calculate


FOLDER_SIZE = 200
THREAD_NUM = 200
MAX_NUM = 40000
thread_limiter = threading.Semaphore(THREAD_NUM)


def get_save_embedding(data_dict, unified_id, success_id_set, id_dict):
    # Acquire the semaphore before proceeding
    thread_limiter.acquire()
    try:
        # Place your function's logic here
        idx = id_dict[unified_id]
        folder = os.path.join(open_ai_full_admission_embedding_folder, str(idx))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, unified_id + '.pkl')

        success_flag = False
        if os.path.exists(path):
            pickle.load(open(path, 'rb'))
            success_flag = True
        if not success_flag:
            english = data_dict[unified_id]
            english_embedding = call_open_ai_embedding(english)
            pickle.dump(english_embedding, open(path, 'wb'))
        success_id_set.add(unified_id)
        logger.info('key: {} success'.format(unified_id))
    finally:
        # Release the semaphore when the task is done
        thread_limiter.release()



def read_reserved_note(path):
    data_dict = dict()
    with (open(path, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            unified_id, note_id = line[0], line[3]
            history = ('Complaint: {}\n\n Major Surgical or Invasive Procedure: {}\n\n'
                       'History of Present Illness: {}\n\n Past Medical History: {}\n'
                       ).format(line[7], line[8], line[9], line[10])
            data_dict[unified_id] = history
    return data_dict


def get_next_batch_id(data_dict, transformed_id_dict, id_dict, batch_size):
    id_list = []
    for key in id_dict:
        if key in data_dict and key not in transformed_id_dict:
            id_list.append(key)
        if len(id_list) >= batch_size:
            break
    return id_list


def generate_embedding(data_dict):
    id_seq_list = list(data_dict.keys())
    random.Random(715).shuffle(id_seq_list)
    id_dict = dict()
    for i, unified_id in enumerate(id_seq_list):
        id_dict[unified_id] = i // FOLDER_SIZE

    count_id = 0
    success_id_set = set()
    while count_id < len(id_seq_list):
        start_time = datetime.now()
        unified_ids = get_next_batch_id(data_dict, success_id_set, id_dict, 400)
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
        count_id = len(success_id_set)
        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Time difference: {time_diff}")
        print(f"current embedding dict length: {len(success_id_set)}")


def read_embedding(embedding_folder):
    embedding_dict = dict()
    folder_list = os.listdir(embedding_folder)
    for folder in folder_list:
        folder_path = os.path.join(embedding_folder, folder)
        path_list = os.listdir(folder_path)
        for path in path_list:
            file_path = os.path.join(folder_path, path)
            base_name = os.path.basename(file_path)[:-4]
            embedding_dict[base_name] = pickle.load(open(file_path, 'rb'))
    return embedding_dict


def main():
    origin_data = read_reserved_note(reserved_note_path)
    generate_embedding(origin_data)
    diagnosis_structured_dict, diagnosis_group_dict, group_index_dict = get_structured_diagnosis(reserved_note_path)
    embedding_dict = read_embedding(open_ai_full_admission_embedding_folder)

    data_dict = dict()
    for key in embedding_dict:
        feature = np.array(embedding_dict[key])
        label = np.argmax(diagnosis_structured_dict[key])
        data_dict[key] = label, feature

    key_list = list(data_dict.keys())
    random.Random().shuffle(key_list)

    diagnosis_list_train = []
    observation_list_train = []
    diagnosis_list_test = []
    observation_list_test = []

    train_size = math.ceil(len(key_list) * 0.7)
    for i, key in enumerate(key_list):
        label, feature = data_dict[key]
        if i < train_size:
            diagnosis_list_train.append(label)
            observation_list_train.append(feature)
        else:
            diagnosis_list_test.append(label)
            observation_list_test.append(feature)

    diagnosis_list_train = np.array(diagnosis_list_train)
    observation_list_train = np.array(observation_list_train)
    diagnosis_list_test = np.array(diagnosis_list_test)
    observation_list_test = np.array(observation_list_test)

    max_iter = 500
    clf = MLPClassifier(
        hidden_layer_sizes=[128, 64],
        # random_state=random_state,
        max_iter=max_iter,
        learning_rate_init=0.0001,
        learning_rate='adaptive',
        batch_size=len(diagnosis_list_train) // 50,
    )
    logger.info('start training')
    for i in range(max_iter):
        clf.partial_fit(observation_list_train, diagnosis_list_train, classes=np.unique(diagnosis_list_train))
        if (i + 1) % 50 == 0 or i == max_iter - 1:
            train_diagnosis_prob = clf.predict_proba(observation_list_train)
            test_diagnosis_prob = clf.predict_proba(observation_list_test)
            for top_k in [1, 3, 5, 10, 20, 30, 50]:
                test_top_k_hit = top_calculate(diagnosis_list_test, test_diagnosis_prob, top_k, 98)
                train_top_k_hit = top_calculate(diagnosis_list_train, train_diagnosis_prob, top_k, 98)
                logger.info('iter: {:4d}, train top {:2d} hit: {:5f},'
                            ' test top {:2d} hit: {:5f}'
                            .format(i + 1, top_k, train_top_k_hit, top_k, test_top_k_hit))
            logger.info('\n')


if __name__ == "__main__":
    main()
