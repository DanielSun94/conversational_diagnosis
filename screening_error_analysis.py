import os.path
from screen_agent.read_data import read_data
import numpy as np
from config import screen_folder
import json


def read_conversation_result(folder_path, max_round):
    conversation_result_dict = dict()
    file_list = os.listdir(folder_path)
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        json_str = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        key = '-'.join(json_str['unified_id'].strip().split('-')[1:])
        doctor_observation = json_str['doctor_observation']
        conversation_result_dict[key] = doctor_observation[0:2151]

        not_hit = doctor_observation[:2151][0::3]
        hit_count = int(717 - np.sum(not_hit))
        if hit_count != max_round + 1:
            print(hit_count)
    return conversation_result_dict


def read_oracle_data(folder_path, key_set, minimum_symptom=1, mode='english', read_from_cache=True):
    train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict = \
        read_data(folder_path, minimum_symptom, mode, read_from_cache)
    oracle_data_dict = dict()
    for dataset in valid_dataset, test_dataset:
        for i in range(len(dataset)):
            key, symptom, diagnosis, embedding = dataset[i]
            if key in key_set:
                oracle_data_dict[key] = symptom
    return oracle_data_dict


def main():
    doctor_llm = 'llama2-70b'
    doctor_type = 'ppo'
    max_round = 10
    folder_path = os.path.join(screen_folder, doctor_llm, doctor_type, str(max_round))
    conversation_result_dict = read_conversation_result(folder_path, max_round)
    key_set = set(conversation_result_dict.keys())
    oracle_data_dict = read_oracle_data(folder_path, key_set)

    confusion_mat_dict = dict()
    for key in oracle_data_dict:
        oracle_result = np.array(oracle_data_dict[key][2::3])
        observation_result = np.array(conversation_result_dict[key][2::3])
        known_hit = 1 - np.array(conversation_result_dict[key][0::3])

        tp = np.sum(oracle_result * observation_result * known_hit)
        tn = np.sum((1-oracle_result) * (1-observation_result) * known_hit)
        fp = np.sum((1-oracle_result) * observation_result * known_hit)
        fn = np.sum(oracle_result * (1-observation_result) * known_hit)
        recall = tp / (tp + fn)
        acc = (tp+tn) / (tp+tn+fp+fn)
        precision = tp / (tp + fp)
        f1 = 2 * recall * precision / (recall + precision)
        confusion_mat_dict[key] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1
        }

    avg_acc, avg_recall, avg_precision, avg_f1 = [], [], [], []
    for key in oracle_data_dict:
        if isinstance(confusion_mat_dict[key]['acc'], float) and 0 <= confusion_mat_dict[key]['acc'] <= 1:
            avg_acc.append(confusion_mat_dict[key]['acc'])
        if isinstance(confusion_mat_dict[key]['recall'], float) and 0 <= confusion_mat_dict[key]['recall'] <= 1:
            avg_recall.append(confusion_mat_dict[key]['recall'])
        if isinstance(confusion_mat_dict[key]['precision'], float) and 0 <= confusion_mat_dict[key]['precision'] <= 1:
            avg_precision.append(confusion_mat_dict[key]['precision'])
        if isinstance(confusion_mat_dict[key]['f1'], float) and 0 <= confusion_mat_dict[key]['f1'] <= 1:
            avg_f1.append(confusion_mat_dict[key]['f1'])

    print('doctor_llm: {}'.format(doctor_llm))
    print('doctor_type: {}'.format(doctor_type))
    print('max_round: {}'.format(max_round))
    print('data size: {}'.format(len(oracle_data_dict)))
    print('avg acc: {}'.format(np.average(avg_acc)))
    print('avg recall: {}'.format(np.average(avg_recall)))
    print('avg precision: {}'.format(np.average(avg_precision)))
    print('avg f1: {}'.format(np.average(avg_f1)))


if __name__ == '__main__':
    main()
