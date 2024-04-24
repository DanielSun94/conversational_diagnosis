import os
import json
import math
import random
import copy
from patient_simulator_util import process_full_info_data
from differential_patient_filter import patient_filter


class BatchGeneration(object):
    def __init__(self, positive_list, negative_list, positive_batch_size, negative_batch_size, shuffle=True,
                 random_seed=715):
        self.positive_list = positive_list
        self.negative_list = negative_list
        self.positive_batch_size = positive_batch_size
        self.negative_batch_size = negative_batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.current_positive_list = None
        self.current_negative_list = None
        self.current_positive_idx = None
        self.current_negative_idx = None
        self.reset()

    def reset(self):
        positive_list = copy.deepcopy(self.positive_list)
        negative_list = copy.deepcopy(self.negative_list)
        if self.shuffle:
            random.Random(self.random_seed).shuffle(positive_list)
            random.Random(self.random_seed).shuffle(negative_list)
        self.current_positive_list = positive_list
        self.current_negative_list = negative_list
        self.current_negative_idx = 0
        self.current_positive_idx = 0

    def get_next_batch(self):
        neg_idx, pos_idx = self.current_negative_idx, self.current_positive_idx
        if ((neg_idx + self.negative_batch_size) > len(self.current_negative_list) or
                (pos_idx + self.positive_batch_size) > len(self.current_positive_list)):
            self.reset()
            neg_idx, pos_idx = self.current_negative_idx, self.current_positive_idx
        next_positive = self.current_positive_list[pos_idx: pos_idx + self.positive_batch_size]
        next_negative = self.current_negative_list[neg_idx: neg_idx + self.negative_batch_size]

        self.current_negative_idx += self.negative_batch_size
        self.current_positive_idx += self.positive_batch_size
        return next_positive, next_negative




def get_data(disease, filter_name, full_positive_num, full_negative_num, train_portion, valid_portion,
             batch_positive_num, batch_negative_num, random_seed=715):
    patient_dataset = process_full_info_data()
    positive_list, negative_list = patient_filter(patient_dataset, disease, llm_name=filter_name)
    positive_list = positive_list[:full_positive_num]
    negative_list = negative_list[:full_negative_num]
    random.Random(random_seed).shuffle(positive_list)
    random.Random(random_seed).shuffle(negative_list)

    train_positive_list = positive_list[:math.ceil(len(positive_list) * train_portion)]
    train_negative_list = negative_list[:math.ceil(len(negative_list) * train_portion)]
    valid_positive_list = positive_list[math.ceil(len(positive_list) * train_portion):
                                        math.ceil(len(positive_list) * (train_portion + valid_portion))]
    valid_negative_list = negative_list[math.ceil(len(negative_list) * train_portion):
                                        math.ceil(len(positive_list) * (train_portion + valid_portion))]
    test_positive_list = positive_list[math.ceil(len(positive_list) * (train_portion + valid_portion)):]
    test_negative_list = negative_list[math.ceil(len(negative_list) * (train_portion + valid_portion)):]
    batch_generate = BatchGeneration(train_positive_list, train_negative_list, batch_positive_num, batch_negative_num)
    return (patient_dataset, batch_generate, valid_positive_list, valid_negative_list,
            test_positive_list, test_negative_list)


class GetNextBatch(object):
    def __init__(self, key_list, batch_size):
        self.key_list = key_list
        self.batch_size = batch_size
        self.current_idx = 0
        self.terminate = False

    def next_batch(self):
        if self.terminate:
            return []
        if self.current_idx + self.batch_size < len(self.key_list):
            batch = self.key_list[self.current_idx: self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            return batch, self.terminate
        else:
            batch = self.key_list[self.current_idx:]
            self.terminate = True
            return batch, self.terminate


def evaluate_performance(result_folder):
    path_list = os.listdir(result_folder)
    valid_path_list = []
    for path in path_list:
        info_list = path.strip().split('_')
        assert info_list[2] == 'confirmed' or info_list[2] == 'excluded'
        disease = info_list[0]
        flag = True if info_list[2] == 'confirmed' else False
        key = info_list[3]
        model = "_".join(info_list[4:-1] + [info_list[1]])
        full_path = os.path.join(result_folder, path)
        valid_path_list.append([full_path, key, flag, model, disease])

    result_list = list()
    for item in valid_path_list:
        full_path, key, confirm_flag, model, disease = item
        json_str = json.load(open(full_path, 'r', encoding='utf-8-sig'))
        last_utterance = json_str['dialogue'][-1]['question']
        if confirm_flag:
            if 'you have' in last_utterance.lower() or 'confirm' in last_utterance.lower():
                result_list.append([disease, key, model, 'TP', 'POSITIVE', len(json_str['dialogue'])])
            elif 'too long' in last_utterance.lower():
                result_list.append([disease, key, model, 'FN', 'POSITIVE', len(json_str['dialogue'])])
            else:
                result_list.append([disease, key, model, 'FN', 'POSITIVE', len(json_str['dialogue'])])
        else:
            if 'you don\'t have' in last_utterance.lower() or 'exclude' in last_utterance.lower():
                result_list.append([disease, key, model, 'TN', 'NEGATIVE', len(json_str['dialogue'])])
            elif 'too long' in last_utterance.lower():
                result_list.append([disease, key, model, 'TN', 'NEGATIVE', len(json_str['dialogue'])])
            else:
                result_list.append([disease, key, model, 'FP', 'NEGATIVE', len(json_str['dialogue'])])

    result_dict = dict()
    for item in result_list:
        disease, model, result = item[0], item[2], item[3]
        key = disease + '-' + model
        if key not in result_dict:
            result_dict[key] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, 'FAILED': 0}
        result_dict[key][result] += 1

    for key in result_dict:
        tp, tn, fp, fn, failed = (result_dict[key]["TP"], result_dict[key]["TN"],
                                  result_dict[key]["FP"], result_dict[key]["FN"], result_dict[key]['FAILED'])
        size = tp + tn + fp + fn + failed
        success_rate = (tp + tn + fp + fn) / size if size > 0 else 'na'
        acc = ((tp + tn) / (tp + tn + fp + fn + failed)) if (tp + tn + fp + fn + failed) > 0 else 'na'
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 'na'
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 'na'
        f1 = (2 * precision * recall / (precision + recall)) \
            if (isinstance(recall, float) and isinstance(precision, float)) else 'na'
        print('model: {}, acc: {}, precision: {}, recall: {}, f1: {}, success_rate: {}, size: {}'
              .format(key, acc, precision, recall, f1, success_rate, size))
    print('\n\n\n\n')
    return result_list, result_dict