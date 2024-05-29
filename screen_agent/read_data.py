import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocess.util import get_icd_mapping_dict, get_mapped_diagnosis_icd
import csv
import math
import pickle
import random
from screen_config import (structured_data_folder, prepared_data_pkl_path, reserved_note_path,
                           open_ai_admission_embedding_folder, question_embedding_folder,
                           translation_path, data_fraction_path_template, full_diagnosis_group_path)
from logger import logger
from itertools import islice
import numpy as np
import json
import requests
from torch.utils.data import Dataset


def main():
    mode = 'chinese'
    data_folder = structured_data_folder
    read_data_full_diagnosis(data_folder, 1, mode, True, True)
    # read_data_primary_diagnosis(data_folder, 1, mode, True, True)
    logger.info('success')


def read_data_full_diagnosis(data_folder, minimum_symptom, mode, question, read_from_cache):
    dump = get_symptom_full_diagnosis(data_folder, minimum_symptom, question, read_from_cache)
    split_data = data_split(dump, mode)
    (train_key, valid_key, test_key), (train_symptom, valid_symptom, test_symptom), \
        (train_diag, valid_diag, test_diag), (train_embedding, valid_embedding, test_embedding),_ = split_data
    train_dataset = DiagnosisDataset(train_key, train_symptom, train_diag, train_embedding)
    valid_dataset = DiagnosisDataset(valid_key, valid_symptom, valid_diag, valid_embedding)
    test_dataset = DiagnosisDataset(test_key, test_symptom, test_diag, test_embedding)
    diagnosis_index_dict = dump['diagnosis_group_index_dict']
    symptom_index_dict = dump['symptom_index_dict']
    return train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict


def read_data_primary_diagnosis(data_folder, minimum_symptom, mode, question, read_from_cache):
    dump = get_symptom_primary_diagnosis(data_folder, minimum_symptom, question, read_from_cache)
    split_data = data_split(dump, mode)
    (train_key, valid_key, test_key), (train_symptom, valid_symptom, test_symptom), \
        (train_diag, valid_diag, test_diag), (train_embedding, valid_embedding, test_embedding),_ = split_data
    train_dataset = DiagnosisDataset(train_key, train_symptom, train_diag, train_embedding)
    valid_dataset = DiagnosisDataset(valid_key, valid_symptom, valid_diag, valid_embedding)
    test_dataset = DiagnosisDataset(test_key, test_symptom, test_diag, test_embedding)
    diagnosis_index_dict = dump['diagnosis_group_index_dict']
    symptom_index_dict = dump['symptom_index_dict']
    return train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict


def data_split(dump, mode, read_from_cache=True, random_split_code=1994):
    split_ratio = [0.7, 0.05, 0.25]
    path = data_fraction_path_template.format(random_split_code, split_ratio[0], split_ratio[1], split_ratio[2])
    if os.path.exists(path) and read_from_cache:
        list_dict = pickle.load(open(path, 'rb'))
        train_key_list = list_dict['train_key_list']
        valid_key_list = list_dict['valid_key_list']
        test_key_list = list_dict['test_key_list']
        logger.info('id list dict loaded')
    else:
        fused_data_key_list = [key for key in dump['fused_data']]
        random.Random(random_split_code).shuffle(fused_data_key_list)
        train_key_list = fused_data_key_list[: math.floor(len(fused_data_key_list) * split_ratio[0])]
        valid_key_list = fused_data_key_list[math.floor(len(fused_data_key_list) * split_ratio[0]):
                                             math.floor(len(fused_data_key_list) * (split_ratio[1] + split_ratio[0]))]
        test_key_list = fused_data_key_list[math.floor(len(fused_data_key_list) * (split_ratio[1] + split_ratio[0])):]
        list_dict = {
            'train_key_list': train_key_list,
            'valid_key_list': valid_key_list,
            'test_key_list': test_key_list
        }
        pickle.dump(list_dict, open(path, 'wb'))
        logger.info('id list dict generated')

    train_symptom, train_diag, valid_symptom, valid_diag, test_symptom, test_diag = [], [], [], [], [], []
    train_embedding, valid_embedding, test_embedding = [], [], []
    exist_train_key_list, exist_valid_key_list, exist_test_key_list = [], [], []
    fused_data = dump['fused_data']
    failed_match_count = 0
    for key_list, exist_key_list, symptom_list, diagnosis_list, embedding_list in zip(
            (train_key_list, valid_key_list, test_key_list),
            (exist_train_key_list, exist_valid_key_list, exist_test_key_list),
            (train_symptom, valid_symptom, test_symptom),
            (train_diag, valid_diag, test_diag),
            (train_embedding, valid_embedding, test_embedding)
    ):
        for key in key_list:
            if key in fused_data:
                exist_key_list.append(key)
                symptom_list.append(fused_data[key]['symptom'])
                diagnosis_list.append(fused_data[key]['diagnosis'])
                embedding_list.append(fused_data[key]['{}_text_embedding'.format(mode)])
            else:
                failed_match_count += 1
                logger.info('key not exist')
    logger.info('failed match count: {}'.format(failed_match_count))
    logger.info('train size: {}, valid size: {}, test size: {}'
                .format(len(train_diag), len(valid_diag),  len(test_diag)))
    return [exist_train_key_list, exist_valid_key_list, exist_test_key_list], \
        [train_symptom, valid_symptom, test_symptom], \
        [train_diag, valid_diag, test_diag], \
        [train_embedding, valid_embedding, test_embedding], dump


def read_embedding(question):
    if question:
        path = question_embedding_folder
    else:
        path = open_ai_admission_embedding_folder
    embedding_dict = dict()
    folder_list = os.listdir(path)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(path, folder))
        for file_name in file_list:
            file_path = os.path.join(path, folder, file_name)
            base_name = os.path.basename(file_path)[:-4]
            embedding = pickle.load(open(file_path, 'rb'))
            embedding_dict[base_name] = embedding
    return embedding_dict


def get_symptom_full_diagnosis(data_folder, minimum_symptom, question, read_from_cache):
    fused_data_path = prepared_data_pkl_path.format(minimum_symptom, '{}_full_diagnosis_fused_data'.format(question))
    info_dict_path = prepared_data_pkl_path.format(minimum_symptom, '{}_full_diagnosis_info_dict'.format(question))
    if read_from_cache and os.path.exists(fused_data_path):
        fused_data_dict = pickle.load(open(fused_data_path, 'rb'))
        info_dict = pickle.load(open(info_dict_path, 'rb'))
        logger.info('read from cache')
    else:
        logger.info('data generation')
        file_path_list = get_file_path_list(data_folder)
        symptom_dict = get_symptom_index_name_dict(file_path_list[0])
        diagnosis_structured_dict, diagnosis_group_dict, group_index_dict = get_structured_diagnosis_v2(code_num=3)
        # chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping = \
        #     data_translate(group_index_dict, symptom_dict, True)

        embedding_dict = read_embedding(question)
        structured_symptom_dict = get_structured_symptom(file_path_list, symptom_dict, minimum_symptom)
        fused_data_dict = dict()
        for key in structured_symptom_dict:
            if key not in embedding_dict or key not in diagnosis_group_dict:
                continue
            fused_data_dict[key] = {
                'symptom': structured_symptom_dict[key],
                'diagnosis': diagnosis_structured_dict[key],
                'chinese_text_embedding': embedding_dict[key]['chinese'],
                'english_text_embedding': embedding_dict[key]['english']
            }

        info_dict = {
            'diagnosis_group_dict': diagnosis_group_dict,
            'diagnosis_group_index_dict': group_index_dict,
            'symptom_index_dict': symptom_dict,
            # 'chinese_diagnosis_group_index_dict': chinese_diagnosis_group_index_dict,
            # 'chinese_symptom_index_dict': chinese_symptom_index_dict,
            # 'diagnosis_mapping': diagnosis_mapping,
            # 'symptom_mapping': symptom_mapping
        }
        pickle.dump(fused_data_dict, open(fused_data_path, 'wb'))
        pickle.dump(info_dict, open(info_dict_path, 'wb'))
    logger.info('full data size: {}'.format(len(fused_data_dict)))

    dump = {
        'fused_data': fused_data_dict,
        'diagnosis_group_dict': info_dict['diagnosis_group_dict'],
        'diagnosis_group_index_dict': info_dict['diagnosis_group_index_dict'],
        'symptom_index_dict': info_dict['symptom_index_dict']
    }
    return dump


def get_symptom_primary_diagnosis(data_folder, minimum_symptom, question, read_from_cache):
    fused_data_path = prepared_data_pkl_path.format(minimum_symptom, '{}_primary_diagnosis_fused_data'.format(question))
    info_dict_path = prepared_data_pkl_path.format(minimum_symptom, '{}_primary_diagnosis_dict'.format(question))

    if read_from_cache and os.path.exists(fused_data_path):
        fused_data_dict = pickle.load(open(fused_data_path, 'rb'))
        info_dict = pickle.load(open(info_dict_path, 'rb'))
        logger.info('read from cache')
    else:
        logger.info('data generation')
        embedding_dict = read_embedding(question)

        file_path_list = get_file_path_list(data_folder)
        symptom_dict = get_symptom_index_name_dict(file_path_list[0])
        structured_symptom_dict = get_structured_symptom(file_path_list, symptom_dict, minimum_symptom)

        diagnosis_structured_dict, diagnosis_group_dict, group_index_dict = \
            get_structured_diagnosis_v1(reserved_note_path)
        # chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping = \
        #     data_translate(group_index_dict, symptom_dict, True)

        fused_data_dict = dict()
        for key in structured_symptom_dict:
            if key not in embedding_dict or key not in diagnosis_structured_dict:
                continue
            fused_data_dict[key] = {
                'symptom': structured_symptom_dict[key],
                'diagnosis': diagnosis_structured_dict[key],
                'chinese_text_embedding': embedding_dict[key]['chinese'],
                'english_text_embedding': embedding_dict[key]['english']
            }

        info_dict = {
            'diagnosis_group_dict': diagnosis_group_dict,
            'diagnosis_group_index_dict': group_index_dict,
            'symptom_index_dict': symptom_dict,
            # # 'chinese_diagnosis_group_index_dict': chinese_diagnosis_group_index_dict,
            # # 'chinese_symptom_index_dict': chinese_symptom_index_dict,
            # 'diagnosis_mapping': diagnosis_mapping,
            # 'symptom_mapping': symptom_mapping
        }
        pickle.dump(fused_data_dict, open(fused_data_path, 'wb'))
        pickle.dump(info_dict, open(info_dict_path, 'wb'))
    logger.info('full data size: {}'.format(len(fused_data_dict)))

    dump = {
        'fused_data': fused_data_dict,
        'diagnosis_group_dict': info_dict['diagnosis_group_dict'],
        'diagnosis_group_index_dict': info_dict['diagnosis_group_index_dict'],
        'symptom_index_dict': info_dict['symptom_index_dict']
    }
    return dump


def data_translate(diagnosis_group_index_dict, symptom_index_dict, read_from_cache):
    chinese_diagnosis_group_index_dict = dict()
    chinese_symptom_index_dict = dict()
    diagnosis_mapping, symptom_mapping = dict(), dict()

    if read_from_cache and os.path.exists(translation_path):
        chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping = \
            pickle.load(open(translation_path, 'rb'))
        return chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping

    logger.info('start diagnosis and symptom translation')
    token = ''
    url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token
    headers = {'Content-Type': 'application/json'}

    for symptom in symptom_index_dict:
        index = symptom_index_dict[symptom][0]
        # 因为翻译机制的原因，部分疼痛有一模一样的翻译,手工修改
        if index == 28:
            chinese_translation = '腹痛，疼痛灼烧'
        elif index == 35:
            chinese_translation = '腹痛，疼痛尖锐'
        else:
            payload = {'q': symptom, 'from': 'en', 'to': 'zh'}
            payload = json.dumps(payload)
            response = requests.request("POST", url, headers=headers, data=payload).json()
            chinese_translation = response['result']['trans_result'][0]['dst']
        if chinese_translation not in chinese_symptom_index_dict:
            chinese_symptom_index_dict[chinese_translation] = \
                index, symptom_index_dict[symptom][1], symptom_index_dict[symptom][2]
        else:
            print('Translate Duplicate')
        symptom_mapping[symptom] = chinese_translation
        symptom_mapping[chinese_translation] = symptom

    for key in diagnosis_group_index_dict:
        index = diagnosis_group_index_dict[key]
        payload = {'q': key, 'from': 'en', 'to': 'zh'}
        payload = json.dumps(payload)
        response = requests.request("POST", url, headers=headers, data=payload).json()
        chinese_translation = response['result']['trans_result'][0]['dst']
        chinese_diagnosis_group_index_dict[chinese_translation] = index
        diagnosis_mapping[key] = chinese_translation
        diagnosis_mapping[chinese_translation] = key

    pickle.dump((chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping),
                open(translation_path, 'wb'))
    return chinese_diagnosis_group_index_dict, chinese_symptom_index_dict, diagnosis_mapping, symptom_mapping


def get_structured_diagnosis_v2(max_seq_num=5, code_num=3):
    """
    将ICD-9映射为ICD-10后处理，考虑所有疾病诊断
    """
    diagnosis_group_dict, code_index_dict, group_index_dict = dict(), dict(), dict()
    icd_dict = get_icd_mapping_dict()
    diagnosis_dict = get_mapped_diagnosis_icd(icd_dict)

    icd_unified_dict = dict()
    with open(full_diagnosis_group_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd_code, _, delete, english, english_group, chinese, chinese_group = line[:7]
            if delete.lower() == 'true':
                continue
            if chinese_group not in group_index_dict:
                group_index_dict[chinese_group] = len(group_index_dict)
            if icd_code not in code_index_dict:
                code_index_dict[icd_code] = group_index_dict[chinese_group]
            icd_unified_dict[icd_code] = [english, english_group, chinese, chinese_group]

    for patient_id in diagnosis_dict:
        for visit_id in diagnosis_dict[patient_id]:
            unified_id = patient_id + "-" + visit_id
            diagnosis_group_dict[unified_id] = dict()
            for seq_num in diagnosis_dict[patient_id][visit_id]:
                icd_code = diagnosis_dict[patient_id][visit_id][seq_num]
                icd_code_abbr = icd_code[: code_num].lower()
                if icd_code_abbr not in icd_unified_dict:
                    continue
                group_index = code_index_dict[icd_code_abbr]
                if icd_code not in diagnosis_group_dict[unified_id]:
                    diagnosis_group_dict[unified_id][icd_code_abbr] = group_index, int(seq_num)
                else:
                    previous_seq_num = diagnosis_group_dict[unified_id][icd_code]
                    if previous_seq_num > int(seq_num):
                        diagnosis_group_dict[unified_id][icd_code_abbr] = group_index, int(seq_num)

    group_index_list = [[key, group_index_dict[key]] for key in group_index_dict]
    group_index_list = sorted(group_index_list, key=lambda x: x[0])
    for i, item in enumerate(group_index_list):
        logger.info(item + [str(i + 1)])

    # 只考虑前10，前十rank之后的直接丢弃。用等差数列进行评分；为了使评分主要在softmax的梯度区间，取-4到5的值
    diagnosis_structured_dict = dict()
    for unified_id in diagnosis_group_dict:
        diagnosis_structured = np.zeros(len(group_index_dict))
        for icd_code in diagnosis_group_dict[unified_id]:
            group_index, seq_num = diagnosis_group_dict[unified_id][icd_code]
            if seq_num > max_seq_num:
                continue
            diagnosis_structured[group_index] = 1
        diagnosis_structured_dict[unified_id] = diagnosis_structured
    return diagnosis_structured_dict, diagnosis_group_dict, group_index_dict



def get_structured_diagnosis_v1(emr_path):
    """
    即一开始的版本，ICD-9和ICD-10分开处理，只考虑primary diagnosis
    """
    diagnosis_group_dict, group_index_dict = dict(), dict()
    with open(emr_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            unified_id = line[0]
            diagnosis_group = line[4]
            diagnosis_description = line[5]
            if diagnosis_group not in group_index_dict:
                group_index_dict[diagnosis_group] = len(group_index_dict)
            index = group_index_dict[diagnosis_group]
            diagnosis_group_dict[unified_id] = [index, diagnosis_group, diagnosis_description]

    diagnosis_structured_dict = dict()
    for unified_id in diagnosis_group_dict:
        diagnosis_structured = np.zeros(len(group_index_dict))
        diagnosis_structured[diagnosis_group_dict[unified_id][0]] = 1
        diagnosis_structured_dict[unified_id] = diagnosis_structured

    group_index_list = [[key, group_index_dict[key]] for key in group_index_dict]
    group_index_list = sorted(group_index_list, key=lambda x: x[0])
    for i, item in enumerate(group_index_list):
        logger.info(item + [str(i+1)])
    return diagnosis_structured_dict, diagnosis_group_dict, group_index_dict


def get_structured_symptom(file_path_list, symptom_dict, minimum_symptom):
    data_dict = dict()
    total_symptom_count = 0
    reserve_symptom_count = 0
    for file_path in file_path_list:
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            positive_symptom_count = 0
            sample_data = np.zeros(len(symptom_dict) * 3)
            visit_key = '-'.join(os.path.basename(file_path).strip().split('-')[: 2])
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                symptom, factor_group, factor, state = line
                assert state == 'YES' or state == 'NO' or state == "NA"
                if state == 'YES':
                    state_num = 2
                    positive_symptom_count += 1
                    total_symptom_count += 1
                elif state == 'NO':
                    state_num = 1
                else:
                    state_num = 0

                if factor_group == 'N/A':
                    key = symptom
                else:
                    key = symptom + ', ' + factor_group + " " + factor

                index = symptom_dict[key][0]
                sample_data[index*3+state_num] = 1

            if positive_symptom_count > 0:
                # 只要有positive的，则28个first level里必须有1
                count = 0
                for i in range(28):
                    count += sample_data[i*3+2]
                assert count > 0
            if positive_symptom_count >= minimum_symptom:
                data_dict[visit_key] = sample_data
                reserve_symptom_count += positive_symptom_count
    logger.info('avg symptom per visit: {}'.format(total_symptom_count/len(file_path_list)))
    logger.info('avg reserve symptom per visit: {}'.format(reserve_symptom_count / len(data_dict)))
    return data_dict


def get_symptom_index_name_dict(file_path):
    symptom_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, factor_group, factor, state = line
            if factor_group == 'N/A':
                level = 1
                if symptom not in symptom_dict:
                    symptom_dict[symptom] = [len(symptom_dict), level, None]
            else:
                level = 2
                key = symptom + ', ' + factor_group + " " + factor
                relate_index = symptom_dict[symptom][0]
                symptom_dict[key] = [len(symptom_dict), level, relate_index]
    return symptom_dict


def get_file_path_list(data_folder):
    file_path_list = list()
    folder_list = os.listdir(data_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(data_folder, folder))
        for file in file_list:
            if 'symptom.csv' in file:
                path = os.path.join(data_folder, folder, file)
                file_path_list.append(path)
    return file_path_list


class DiagnosisDataset(Dataset):
    def __init__(self, key, symptom, diagnosis, embedding):
        self.key = key
        self.symptom = symptom
        self.embedding = embedding
        self.diagnosis = diagnosis

    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        key = self.key[idx]
        symptom = self.symptom[idx]
        diagnosis = self.diagnosis[idx]
        embedding = self.embedding[idx]
        return key, symptom, diagnosis, embedding


if __name__ == '__main__':
    main()
