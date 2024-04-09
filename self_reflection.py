import copy
import math
import os
import json
import random
from patient_simulator import PatientSimulator
from patient_simulator_util import process_full_info_data
from differential_patient_filter import patient_filter
from doctor_simulator import (DoctorSimulator, parse_questions, read_text)
from llm_util import call_open_ai
from environment import Environment
from datetime import datetime
from config import logger, self_reflection_folder, diagnosis_procedure_template_path
import threading


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


def run(dataset, positive_list, negative_list, batch_idx, doctor_llm_name, patient_llm_name, procedure_structure,
        start_index, disease, result_folder, flag_2):
    batch_data_dict = dict()
    for (flag_1, data_list) in (
            zip(('confirmed', 'excluded'), (positive_list, negative_list))):
        for key in data_list:
            data = dataset[key][0]
            path = os.path.join(result_folder, '{}_reflect_{}_{}_{}_{}_{}.json'
                                .format(disease, batch_idx, flag_1, key, doctor_llm_name, flag_2))
            batch_data_dict['-'.join([key, doctor_llm_name, flag_1, flag_2])] = [flag_1, key, data, path, flag_2]

    key_list = sorted(list(batch_data_dict.keys()))
    terminate = False
    id_batch = GetNextBatch(key_list, 20)
    while not terminate:
        threads = []
        batch, terminate = id_batch.next_batch()
        for key in batch:
            flag_1, key, data, path, flag_2 = batch_data_dict[key]

            # env patient和doctor都和线程绑定
            patient = PatientSimulator(patient_llm_name)
            ka_doctor = DoctorSimulator(procedure_structure, start_index, disease, doctor_llm_name)
            env = Environment(patient, ka_doctor)

            thread = threading.Thread(
                target=run_sample,
                args=(path, env, key, data))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


def run_sample(path, env, key, data):
    if not os.path.exists(path):
        result = env.run(key, data)
        result_str = json.dumps(result)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(result_str)


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


def generate_reflection(data, result_folder, batch_index, result_list, knowledge_text, knowledge_procedure, disease):
    reflection_list = list()
    for item in result_list:
        key, result = item[0], item[2]
        assert result in {'TP', "TN", "FP", "FN"}
        if result == 'TP' or result == 'TN':
            continue
        if result == 'FP':
            false_str = 'false positive'
        else:
            false_str = 'false negative'
        path_list = os.listdir(result_folder)
        valid_path_list = []
        for path in path_list:
            info_list = path.strip().split('_')
            batch = int(info_list[1])
            result_key = info_list[3]
            if batch == batch_index and key == result_key:
                valid_path_list.append(os.path.join(result_folder, path))
        assert len(valid_path_list) == 1
        json_str = json.load(open(valid_path_list[0], 'r', encoding='utf-8-sig'))
        dialogue = json_str['dialogue']
        dialogue_str = ''
        for i, turn in enumerate(dialogue):
            dialogue_str += ('Round: {}\nQuestion: {}\nAnswer:{}\n\n'
                             .format(i + 1, turn['question'], turn['response']))

        emr = data[key][0]
        prompt = (("Please assume that you are a senior physician. "
                   "You have summarized a structured {} diagnostic process (Attachment 1) from the clinical guidelines "
                   "(Attachment 2). When diagnosing a patient according to this diagnostic process, "
                   "a {} error occurred. The patient's EMR is shown as follows (Attachment 3), "
                   "and the conversation is shown as Attachment 4. Please think carefully and summarize "
                   "which question (response the number and question content) occurred in the "
                   "structured diagnosis procedure. Please answer briefly.\n\n "
                   "Attachment 1: {}\n\n Attachment 2: {}\n\n Attachment 3: {}\n\n Attachment 4: {}")
                  .format(disease, false_str, knowledge_procedure, knowledge_text, emr, dialogue_str))
        response = call_open_ai(prompt, 'gpt_4_turbo')
        reflection_list.append(response)
    return reflection_list


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
        if confirm_flag:
            last_utterance = json_str['dialogue'][-1]['question']
            if 'you have' in last_utterance.lower() or 'confirm' in last_utterance.lower():
                result_list.append([disease, key, model, 'TP', 'POSITIVE', len(json_str['dialogue'])])
            else:
                result_list.append([disease, key, model, 'FN', 'POSITIVE', len(json_str['dialogue'])])
        else:
            last_utterance = json_str['dialogue'][-1]['question']
            if 'you don\'t have' in last_utterance.lower() or 'exclude' in last_utterance.lower():
                result_list.append([disease, key, model, 'TN', 'NEGATIVE', len(json_str['dialogue'])])
            else:
                result_list.append([disease, key, model, 'FP', 'NEGATIVE', len(json_str['dialogue'])])

    result_dict = dict()
    for item in result_list:
        disease, model, result = item[0], item[2], item[3]
        key = disease + '-' + model
        if key not in result_dict:
            result_dict[key] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        result_dict[key][result] += 1

    for key in result_dict:
        tp, tn, fp, fn = (result_dict[key]["TP"], result_dict[key]["TN"],
                          result_dict[key]["FP"], result_dict[key]["FN"])
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print('model: {}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'
              .format(key, acc, precision, recall, f1))
    print('\n\n\n\n')
    return result_list, result_dict


def revise_decision_procedure(disease, reflect_list, decision_procedure, decision_text, version_id, path_template):
    reflect_list_str = ''
    for i, reflect in enumerate(reflect_list):
        reflect_list_str += 'Reflect No. {}: {}\n\n\n\n'.format(i, reflect)
    prompt = (("Please assume you are an experienced cardiologist. You have summarized a structured "
               "diagnostic process (Attachment 1) for heart failure based on clinical guidelines (Attachment 2). "
               "Now, it has been found that this diagnostic process has encountered a series of issues in practice, "
               "with reflections found in Attachment 3. Please thoughtfully reconsider and generate a new, "
               "corrected structured diagnostic process. The format of the diagnostic process should adhere "
               "to the following format.\n "
               "(1) Please generate a decision procedure to describe the diagnosis of {} step by step "
               "based on the following clinical guideline information. Think carefully.\n "
               "(2) Each step of the procedure must be a question with \"Yes\" or \"No\" as the possible answers. "
               "Each answer must be mapped to a clear next step, such as jumping to another question or generating a "
               "conclusion and terminating the procedure.\n"
               "(3) If the answer maps to a jump step, the sentence should be like '#PROCEED TO QUESTION #n#', "
               "where all letters are in uppercase.\n"
               "(4) You need to summarize the title of the decision procedure by analyzing the context and the title.\n"
               "(5) The tree should be like: \n"
               "  Title: ...\n"
               "  Start \n"
               "  #QUESTION #1#: ... \n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  #QUESTION #2#: ... \n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  ...\n"
               "  #QUESTION #n#: ...\n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  End\n"
               "(6) Please do not use information that is not included in the given context.\n"
               "(7) The final output of the decision procedure should explicitly express either "
               "\"YOU HAVE {}\" or \"YOU DON'T HAVE {}\". Other final outputs are not allowed.\n"
               "(8) Please make the procedure as precise as possible. It can be long, but it must not contain errors. "
               "(9) Questions are not allowed to have nested structure."
               "Each condition must have \"YES\" or \"NO\" as the two branches; no other branches are allowed.\n"
               "Attachment 1: {}\n\n Attachment 2: {}\n\n Attachment 3: {}")
              .format(disease, disease, disease, decision_procedure, decision_text, reflect_list_str))
    success_flag, start_index, response = False, None, None
    while not success_flag:
        response = call_open_ai(prompt, model_name='gpt_4_turbo')
        decision_procedure, start_index = parse_questions(response)
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(path_template.format(version_id, now), 'w', encoding='utf-8-sig', newline='') as f:
            f.writelines(response)
        success_flag = True
    return decision_procedure, start_index, response


def main():
    experiment_no = 'B'
    disease = "HEART FAILURE"
    patient_llm_name = 'gpt_4_turbo'
    doctor_llm_name = 'gpt_4_turbo'
    filter_name = 'gpt_4_turbo'
    full_positive_num = 160
    full_negative_num = 160
    batch_positive_num = 10
    batch_negative_num = 10
    train_portion = 0.375
    valid_portion = 0
    result_folder = os.path.join(self_reflection_folder, experiment_no)
    os.makedirs(result_folder, exist_ok=True)
    decision_procedure_path_dict = \
        {disease: diagnosis_procedure_template_path.format(2)}
    procedure_structure, start_index, procedure_text = read_text(decision_procedure_path_dict[disease])
    # with open(knowledge_origin_text_path, 'r', encoding='utf-8-sig') as f:
    #     diagnosis_text = '\n'.join(f.readlines())

    (dataset, batch_generate, valid_positive_list, valid_negative_list, test_positive_list, test_negative_list) = \
        get_data(disease, filter_name, full_positive_num, full_negative_num, train_portion, valid_portion,
                 batch_positive_num, batch_negative_num)

    logger.info('start analyze')

    target_idx = 2
    for i in range(6):
        batch_positive, batch_negative = batch_generate.get_next_batch()
        if i == target_idx:
            run(dataset, batch_positive, batch_negative, i, doctor_llm_name, patient_llm_name, procedure_structure,
                start_index, disease, result_folder, 'train')
            result_list, _ = evaluate_performance(result_folder)


if __name__ == '__main__':
    main()
