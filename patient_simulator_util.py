from config import radiology_note_path, discharge_note_path, reserve_key_path, cache_folder, patient_full_info_path, \
    patient_info_folder, diagnosis_path
from logger import logger
from itertools import islice
import csv
import json
import os
import pickle
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer


def process_full_info_data(read_from_cache=True, stat_data_flag=False):
    if os.path.exists(patient_full_info_path) and read_from_cache:
        data_str_dict = pickle.load(open(patient_full_info_path, 'rb'))
    else:
        reserve_key_set = get_reserve_key(reserve_key_path)
        diagnosis_dict = read_diagnosis(diagnosis_path)
        data_dict = read_data(discharge_note_path, radiology_note_path, reserve_key_set)
        fused_data_dict = fusion_data(data_dict, diagnosis_dict)
        data_str_dict = convert_to_string(fused_data_dict)

        key_list = list(data_str_dict.keys())
        for i, key in enumerate(key_list):
            folder_num = str(i // 400)
            target_folder = os.path.join(patient_info_folder, folder_num)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
            path = os.path.join(target_folder, str(i) + '-' + key + '_emr.txt')
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(data_str_dict[key][0])
            path = os.path.join(target_folder, str(i) + '-' + key + '_diagnosis.txt')
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(data_str_dict[key][1])

        with open(patient_full_info_path, 'wb') as f:
            pickle.dump(data_str_dict, f)
    if stat_data_flag:
        data_statistic_info(data_str_dict)
    return data_str_dict


def read_diagnosis(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            subject_id, hadm_id, seq_num, icd_code, icd_version = line
            if subject_id not in data_dict:
                data_dict[subject_id] = dict()
            if hadm_id not in data_dict[subject_id]:
                data_dict[subject_id][hadm_id] = dict()
            data_dict[subject_id][hadm_id][seq_num] = {'code': icd_code, 'version': icd_version}
    return data_dict


def data_statistic_info(data_str_dict):
    token_num_list, word_num_list = [], []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B")
    logger.info('start encoding')
    count = 0
    for key in data_str_dict:
        content = data_str_dict[key][0]
        word_num = len(content.split(' '))
        content_tokens = tokenizer(content)
        content_tokens = content_tokens.data['input_ids']
        token_num_list.append(len(content_tokens))
        word_num_list.append(word_num)
        count += 1
        if count % 100 == 0:
            logger.info('count: {}'.format(count))

    token_num_list = sorted(token_num_list)
    word_num_list = sorted(word_num_list)
    quad = len(token_num_list) // 4
    logger.info('average word: {}, token: {}'.format(np.mean(word_num_list), np.mean(token_num_list)))
    logger.info('median word: {}, token: {}'.format(np.median(word_num_list), np.median(token_num_list)))
    logger.info('max word: {}, token: {}'.format(np.max(word_num_list), np.max(token_num_list)))
    logger.info('minimum word: {}, token: {}'.format(np.min(word_num_list), np.min(token_num_list)))
    logger.info('25 percentile word: {}, token: {}'.format(word_num_list[quad], token_num_list[quad]))
    logger.info('75 percentile word: {}, token: {}'.format(word_num_list[quad * 3], token_num_list[quad * 3]))


def convert_to_string(fused_data_dict):
    data_str_dict = dict()
    note_template = 'NOTE COMPLETE TIME: {}. \n CONTENT: \n{}\n\n'
    radiology_template = 'EXAM NUMBER: {}\nEXAM TIME: {}. \nCONTENT: \n{}\n\n'
    for key in fused_data_dict:
        store_time, text = fused_data_dict[key]['note']['store_time'], fused_data_dict[key]['note']['text']
        diagnosis_dict = fused_data_dict[key]['diagnosis']
        diagnosis_dict_str = json.dumps(diagnosis_dict)
        radiology_list = []
        for radio_key in fused_data_dict[key]['radiology']:
            start_time, report = fused_data_dict[key]['radiology'][radio_key]['chart_time'], \
                fused_data_dict[key]['radiology'][radio_key]['text']
            radiology_list.append([start_time, report])
        radiology_list = sorted(radiology_list, key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))

        note_text = note_template.format(store_time, text)
        radiology_text = ""
        for i, content in enumerate(radiology_list):
            start_time, report = content
            radiology_text += radiology_template.format(i, start_time, report)

        full_note_text = note_text + radiology_text
        data_str_dict[key] = full_note_text, diagnosis_dict_str
    return data_str_dict


def fusion_data(data_dict, diagnosis_dict):
    radiology_dict, note_dict = data_dict
    fused_data_dict = dict()
    count = 0
    for subject_id in note_dict:
        for hadm_id in note_dict[subject_id]:
            if subject_id not in diagnosis_dict or hadm_id not in diagnosis_dict[subject_id]:
                continue

            count += 1
            assert len(note_dict[subject_id][hadm_id]) == 1
            if subject_id in radiology_dict and hadm_id in radiology_dict[subject_id]:
                radiology_data = radiology_dict[subject_id][hadm_id]
            else:
                radiology_data = {}
            note_id = list(note_dict[subject_id][hadm_id].keys())[0]
            note_data = note_dict[subject_id][hadm_id][note_id]
            key = subject_id + '-' + hadm_id
            fused_data_dict[key] = dict()
            fused_data_dict[key]['note'] = note_data
            fused_data_dict[key]['radiology'] = radiology_data
            fused_data_dict[key]['diagnosis'] = diagnosis_dict[subject_id][hadm_id]
    logger.info('count: {}'.format(count))
    return fused_data_dict


def read_data(note_path, radio_path, reserve_key_set):
    cache_path_1 = os.path.join(cache_folder, 'raw_cache.pkl')
    if os.path.exists(cache_path_1):
        radiology_data, note_data = pickle.load(open(cache_path_1, 'rb'))
    else:
        radiology_data = read_note_data(radio_path, reserve_key_set)
        logger.info('radiology data size: {}'.format(len(radiology_data)))
        note_data = read_note_data(note_path, reserve_key_set)
        logger.info('note data size: {}'.format(len(note_data)))
        pickle.dump([radiology_data, note_data], open(cache_path_1, 'wb'))
    logger.info('success')
    return radiology_data, note_data


def get_reserve_key(key_path):
    reserve_key_set = set()
    with open(key_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            key = line[0]
            unique_id = '-'.join(key.strip().split('-')[0: 2])
            reserve_key_set.add(unique_id)
    return reserve_key_set


def read_note_data(note_path, reserve_key_set):
    reserve_data_dict = dict()
    with open(note_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            note_id, subject_id, hadm_id, note_type, note_seq, chart_time, store_time, text = line
            key = subject_id + "-" + hadm_id
            if key not in reserve_key_set:
                continue

            if subject_id not in reserve_data_dict:
                reserve_data_dict[subject_id] = dict()
            if hadm_id not in reserve_data_dict[subject_id]:
                reserve_data_dict[subject_id][hadm_id] = dict()
            assert note_id not in reserve_data_dict[subject_id][hadm_id]
            reserve_data_dict[subject_id][hadm_id] = {
                "note_type": note_type,
                'note_seq': note_seq,
                "chart_time": chart_time,
                "store_time": store_time,
                "text": text
            }
    return reserve_data_dict
