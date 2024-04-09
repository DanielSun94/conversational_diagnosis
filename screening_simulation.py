import os
import csv
from itertools import islice
import random
import torch
import json
import pickle
from llm_util import call_llm
from config import code_mapping_path, patient_info_folder, patient_info_group_path, screen_folder, reserve_note_file, \
    embedding_folder, screening_planner_path_dict, logger, standard_question_path
from gymnasium.spaces import Box, Discrete
from screnning_agent.config import prepared_data_pkl_path, structured_data_folder, data_fraction_path_template
from environment import Environment
from patient_simulator import PatientSimulator
from screnning_agent.policy_model import SymptomInquireActorCriticPolicy
import re
import numpy as np
from screnning_agent.util import LinearSchedule
import threading


def main():
    eval_mode = False
    init_idx = 0
    data_size = 300
    split_num = 4

    max_round = 20
    mode = 'dialogue'
    patient_llm_name = 'gpt_4_turbo'
    doctor_llm_name = 'llama2-70b'  # gpt_4_turbo llama2-70b gpt_35_turbo
    doctor_type = 'ppo'  # gpt, ppo

    logger.info('init index: {}'.format(init_idx))
    logger.info('thread num (split num): {}'.format(split_num))
    logger.info('data_size: {}'.format(data_size))
    logger.info('max_round: {}'.format(max_round))
    logger.info('mode: {}'.format(mode))
    logger.info('patient_llm_name: {}'.format(patient_llm_name))
    logger.info('doctor_llm_name: {}'.format(doctor_llm_name))
    logger.info('doctor_type: {}'.format(doctor_type))

    path = data_fraction_path_template.format(715, 0.7, 0.05, 0.25)
    assert os.path.exists(path)
    train_id_set = set(pickle.load(open(path, 'rb'))['train_key_list'])

    group_icd_code_dict, icd_code_group_dict = get_icd_mapping_code(code_mapping_path)
    data_dict = read_data(reserve_note_file, patient_info_folder, icd_code_group_dict, patient_info_group_path)

    assert data_size % split_num == 0
    index_list = []
    for i in range(split_num):
        split_size = data_size / split_num
        statr_idx = int(i * split_size) + init_idx
        end_idx = int((i + 1) * split_size) + init_idx
        index_list.append([statr_idx, end_idx])
    key_list = sorted([key for key in data_dict])
    random.Random(715).shuffle(key_list)
    target_key_list = []
    for key in key_list:
        if key not in train_id_set:
            target_key_list.append(key)
        if len(target_key_list) >= data_size+init_idx:
            break

    if not eval_mode:
        screen_planner_path = screening_planner_path_dict[max_round]['planner']
        screen_classifier_path = screening_planner_path_dict[max_round]['classifier']
        embedding_path_dict = read_embedding_path(embedding_folder)
        classifier, policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict = (
            read_planner_classifier(screen_planner_path, screen_classifier_path))
        symptom_path_dict = read_symptom_dict(structured_data_folder)
        index_question_dict = get_translation_dict(standard_question_path, symptom_index_dict, 'english')

        threads = []
        for i, item in enumerate(index_list):
            start_index, end_index = item
            thread = threading.Thread(
                target=eval_thread,
                args=(start_index, end_index, mode, target_key_list, doctor_llm_name, doctor_type, max_round,
                      data_dict, group_icd_code_dict, embedding_path_dict, symptom_path_dict, classifier,
                      policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict, patient_llm_name,
                      index_question_dict))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    else:
        if mode == 'dialogue':
            parse_dialogue_result(os.path.join(screen_folder, doctor_llm_name, doctor_type, str(max_round)),
                                  target_key_list)
        else:
            assert mode == 'review'
            parse_review_result(os.path.join(screen_folder, doctor_llm_name), target_key_list)


def get_translation_dict(path, index_name_dict, language):
    question_dict = dict()
    with open(path, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            key = line[0]
            chinese = line[4]
            english = line[5]
            question_dict[key] = {'chinese': chinese, "english": english}

    key_set_1 = set(index_name_dict.keys())
    key_set_2 = set(question_dict.keys())
    assert len(key_set_2.union(key_set_1)) == len(key_set_2)

    index_question_dict = dict()
    for key in index_name_dict:
        index = index_name_dict[key][0]
        index_question_dict[index] = question_dict[key][language]
    return index_question_dict


def eval_thread(start_index, end_index, mode, target_key_list, doctor_llm_name, doctor_type, max_round,
                data_dict, group_icd_code_dict, embedding_path_dict, symptom_path_dict, classifier,
                policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict, patient_llm_name,
                index_question_dict):
    for index in range(start_index, end_index):
        if mode == 'dialogue':
            key = target_key_list[index]
            target_folder = os.path.join(screen_folder, doctor_llm_name, doctor_type, str(max_round))
            os.makedirs(target_folder, exist_ok=True)
            file_path = os.path.join(str(target_folder), str('{}_dialogue.json'.format(key)))
            full_admission, partial_admission = data_dict[key][0], data_dict[key][1]
            if doctor_type == 'gpt':
                doctor = ScreenPureLLMDoctorSimulator(partial_admission, max_round, doctor_llm_name,
                                                      group_icd_code_dict)
            else:
                assert doctor_type == 'ppo'
                patient_key = '-'.join(key.strip().split('-')[1:])
                if patient_key not in embedding_path_dict or patient_key not in symptom_path_dict:
                    logger.info('patient {} info miss'.format(patient_key))
                    continue
                doctor = ScreeningExternalPlannerDoctorSimulator(
                    classifier, policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict,
                    index_question_dict, embedding_path_dict, symptom_path_dict, max_round, doctor_llm_name,
                    patient_key)
            dialogue_mode(file_path, key, data_dict, doctor, full_admission, patient_llm_name)
        else:
            assert mode == 'review'
            review_mode(index, target_key_list, data_dict, doctor_llm_name, group_icd_code_dict)


def read_symptom_dict(symptom_folder):
    symptom_path_dict = dict()
    folder_list = os.listdir(symptom_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(symptom_folder, folder))
        for file in file_list:
            path = os.path.join(symptom_folder, folder, file)
            if 'symptom' not in file:
                continue
            key = '-'.join(file.strip().split('-')[0:2])
            symptom_path_dict[key] = path
    return symptom_path_dict


def read_planner_classifier(planner_path, clf_path, symptom_num=717, embedding_size=3072):
    classifier, symptom_index_dict = pickle.load(open(clf_path, 'rb'))
    info_dict_path = prepared_data_pkl_path.format('1', 'info_dict')
    info_dict = pickle.load(open(info_dict_path, 'rb'))
    diagnosis_dict = info_dict['diagnosis_group_dict']
    diagnosis_index_mapping = info_dict['diagnosis_group_index_dict']

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 128, 128], vf=[64]),
        symptom_index_dict=symptom_index_dict,
        symptom_num=symptom_num
    )
    policy_weight = torch.load(planner_path, map_location=torch.device('cpu'))
    action_space = Discrete(symptom_num)
    observation_space = Box(low=-10000, high=10000, shape=[symptom_num * 3 + 1 + embedding_size])
    policy_model = SymptomInquireActorCriticPolicy(
        action_space=action_space,
        observation_space=observation_space,
        lr_schedule=LinearSchedule(0.001),
        **policy_kwargs,
    )
    policy_model.load_state_dict(policy_weight)
    return classifier, policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict


def read_embedding_path(general_folder):
    embedding_path_dict = dict()
    folder_list = os.listdir(general_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(general_folder, folder))
        for file in file_list:
            path = os.path.join(general_folder, folder, file)
            key = file[:-4]
            assert key not in embedding_path_dict
            embedding_path_dict[key] = path
    return embedding_path_dict


def parse_review_result(folder, target_key_list):
    file_list = os.listdir(folder)
    rank_list = []
    for file in file_list:
        path = os.path.join(folder, file)
        if 'review' not in path:
            continue

        result_json = json.load(open(path, 'r', encoding='utf-8-sig'))
        true_diagnosis = result_json['diagnosis_label']
        diagnosis_prediction_list = result_json['response']
        diagnosis_id_list = result_json['diagnosis_id_list'].strip().split('\n')
        disease_id_dict = dict()

        rank = 100
        try:
            for line in diagnosis_id_list:
                parts = line.split('#')
                key = int(parts[1])
                value = parts[2].strip()
                disease_id_dict[value] = key
            true_diagnosis_id = disease_id_dict[true_diagnosis]
            numbers = [int(num) for num in re.findall(r'#(\d+)#', diagnosis_prediction_list)]
            for i in range(len(numbers)):
                if true_diagnosis_id == numbers[i]:
                    rank = i
                    break
        except Exception as exc:
            logger.info('{} error, exc: {}'.format(file, exc))
        rank_list.append(rank)
    top_1 = np.sum(np.array(rank_list) < 1) / len(rank_list)
    top_3 = np.sum(np.array(rank_list) < 3) / len(rank_list)
    top_5 = np.sum(np.array(rank_list) < 5) / len(rank_list)
    top_10 = np.sum(np.array(rank_list) < 10) / len(rank_list)
    logger.info('top 1: {}\ntop 3: {}\ntop 5: {}\ntop 10: {}\n'.format(top_1, top_3, top_5, top_10))


def parse_dialogue_result(folder, target_key_list):
    target_key_set = set(target_key_list)
    file_list = os.listdir(folder)
    rank_list = []
    diagnosis_count_dict = dict()
    for file in file_list:
        path = os.path.join(folder, file)
        key = file.strip().split('_')[0]
        if key not in target_key_set:
            continue

        if 'dialogue' not in path:
            continue
        result_json = json.load(open(path, 'r', encoding='utf-8-sig'))
        true_diagnosis = result_json['diagnosis_label']
        diagnosis_prediction_list = result_json['dialogue'][-1]['question']
        diagnosis_id_list = result_json['diagnosis_id_list'].strip().split('\n')
        disease_id_dict = dict()
        if true_diagnosis not in diagnosis_count_dict:
            diagnosis_count_dict[true_diagnosis] = 0
        diagnosis_count_dict[true_diagnosis] += 1

        rank = 100
        try:
            for line in diagnosis_id_list:
                parts = line.split('#')
                key = int(parts[1])
                value = parts[2].strip()
                disease_id_dict[value] = key
            true_diagnosis_id = disease_id_dict[true_diagnosis]
            numbers = [int(num) for num in re.findall(r'#(\d+)#', diagnosis_prediction_list)]
            for i in range(len(numbers)):
                if true_diagnosis_id == numbers[i]:
                    rank = i
                    break
        except Exception as exc:
            logger.info('{} error, exc: {}'.format(file, exc))
        rank_list.append(rank)

    top_1 = np.sum(np.array(rank_list) < 1) / len(rank_list)
    top_3 = np.sum(np.array(rank_list) < 3) / len(rank_list)
    top_5 = np.sum(np.array(rank_list) < 5) / len(rank_list)
    top_10 = np.sum(np.array(rank_list) < 10) / len(rank_list)
    logger.info('result size: {}'.format(len(rank_list)))
    logger.info('\ntop 1: {}\ntop 3: {}\ntop 5: {}\ntop 10: {}\n'.format(top_1, top_3, top_5, top_10))


def review_mode(index, target_key_list, data_dict, doctor_llm_name, group_icd_code_dict):
    key = target_key_list[index]
    path = os.path.join(screen_folder, doctor_llm_name, f'{key}_review.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8-sig') as file:
            data_result = json.load(file)
        logger.info('key: {} already success'.format(key))
    else:
        data = data_dict[key][0]
        doctor = ScreenReviewPureGPTDoctorSimulator(doctor_llm_name, group_icd_code_dict)
        response = doctor.parse(data)
        data_result = dict()
        data_result['response'] = response
        data_result['diagnosis_id_list'] = doctor.disease_list_str
        data_result['diagnosis_icd_list'] = data_dict[key][2]
        data_result['full_emr'] = data_dict[key][4]
        data_result['diagnosis_label'] = data_dict[key][3]
        with open(path, 'w', encoding='utf-8-sig') as file:
            json.dump(data_result, file)
    return key, data_result


def dialogue_mode(path, key, data_dict, doctor, full_admission, patient_llm_name):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8-sig') as file:
            data_result = json.load(file)
        logger.info('key: {} already success'.format(key))
    else:
        patient = PatientSimulator(patient_llm_name)
        environment = Environment(patient, doctor)
        data_result = environment.run(key, full_admission)
        data_result['diagnosis_icd_list'] = data_dict[key][2]
        data_result['full_emr'] = data_dict[key][4]
        data_result['diagnosis_label'] = data_dict[key][3]
        data_result['diagnosis_id_list'] = doctor.disease_list_str
        data_result['doctor_observation'] = list(doctor.current_observation)
        with open(path, 'w', encoding='utf-8-sig') as file:
            json.dump(data_result, file)
    return key, data_result


class ScreenReviewPureGPTDoctorSimulator:
    def __init__(self, llm_name, disease_range_dict):
        self.disease_rage_dict = disease_range_dict
        self.llm_name = llm_name
        self.disease_list_str = self.generate_disease_ranking_list()

    def generate_disease_ranking_list(self):
        range_dict = self.disease_rage_dict
        disease_list = []
        for group in range_dict:
            disease_list.append([len(disease_list), group])
        disease_list_str = "".join(["#{}# {}\n".format(item[0], item[1]) for item in disease_list])
        disease_list_str = disease_list_str.replace('\xa0', " ")
        return disease_list_str

    def parse(self, emr):
        prompt = (
            "Please presume you are a doctor. You need to generate diagnosis decision based on the given EMR:\n "
            "EMR:\n {}\n\n\n. You need to create a list of 10 high-risk diseases. The diseases should be numbered "
            "in such a way that a smaller number indicates a "
            "higher risk associated with the disease.\n"
            "The disease candidate list is:\n{}\n."
            "The output format should be like:\n"
            "START:\n"
            "#3#\n"
            "#9#\n"
            "#4#\n"
            "#5#\n"
            "#12#\n"
            "#43#\n"
            "#28#\n"
            "#90#\n"
            "#16#\n"
            "#20#\n"
            "END\n"
            "if you think the disease #3# has the highest risk and #9# is the second highest risk disease, and so on.\n"
            "..."
        ).format(emr, self.disease_list_str).strip()
        return call_llm(self.llm_name, prompt)


class ScreeningExternalPlannerDoctorSimulator(object):
    def __init__(self, classifier, policy_model, diagnosis_dict, diagnosis_index_mapping, symptom_index_dict,
                 index_question_dict, embedding_path_dict, symptom_path_dict, max_round, llm_name, patient_key,
                 symptom_num=717, embedding_size=3072, primary_symptom_num=28):
        self.classifier = classifier
        self.policy_model = policy_model
        self.diagnosis_dict = diagnosis_dict
        self.diagnosis_index_mapping = diagnosis_index_mapping
        self.index_diagnosis_mapping = {diagnosis_index_mapping[key]: key for key in diagnosis_index_mapping}
        self.symptom_index_dict = symptom_index_dict
        self.index_symptom_dict = {symptom_index_dict[key][0]: key for key in symptom_index_dict}
        self.embedding_path_dict = embedding_path_dict
        self.symptom_path_dict = symptom_path_dict
        self.max_round = max_round
        self.llm_name = llm_name
        self.symptom_num = symptom_num
        self.embedding_size = embedding_size
        self.patient_key = patient_key
        self.index_question_dict = index_question_dict
        self.disease_list_str = self.generate_disease_ranking_list()

        self.terminate = False
        self.conclusion = None
        self.last_action = None
        self.current_observation = None
        self.current_round = 0

    def generate_disease_ranking_list(self):
        disease_list = [[self.diagnosis_index_mapping[key], key] for key in self.diagnosis_index_mapping]
        disease_list_str = "".join(["#{}# {}\n".format(item[0], item[1]) for item in disease_list])
        disease_list_str = disease_list_str.replace('\xa0', " ")
        return disease_list_str

    def reset(self):
        self.terminate = False
        self.last_action = None
        self.conclusion = None
        self.current_round = 0
        self.current_observation = np.zeros(self.symptom_num * 3 + 1 + self.embedding_size)
        self.current_observation[0::3] = 1
        self.current_observation[self.symptom_num * 3] = 0  # 这个值固定为0

        embedding_path = self.embedding_path_dict[self.patient_key]
        symptom_path = self.symptom_path_dict[self.patient_key]
        embedding = np.array(pickle.load(open(embedding_path, 'rb'))['english'])
        self.current_observation[self.symptom_num * 3 + 1:] = embedding

        # 根据设置，至少要告诉ppo一特征
        positive_index_list = []
        with open(symptom_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                symptom, factor_group, factor, state = line
                # 这里只考虑primary info
                if factor != 'N/A':
                    assert factor_group != 'N/A'
                    continue
                assert symptom in self.symptom_index_dict
                assert state == 'YES' or state == 'NO' or state == 'NA'
                if state == 'YES':
                    index = self.symptom_index_dict[symptom][0]
                    assert index < 28
                    positive_index_list.append(index)
        if len(positive_index_list) > 0:
            choice = int(random.choice(positive_index_list))
            self.current_observation[choice * 3 + 2] = 1
            self.current_observation[choice * 3] = 0
        else:
            logger.info('key: {}, no positive first level symptom'.format(self.patient_key))
        logger.info('reset success')

    def run_policy_model(self):
        model_input = torch.unsqueeze(torch.FloatTensor(self.current_observation), 0)
        result = self.policy_model(model_input)
        action = result[0].item()
        action_str = self.index_question_dict[action]
        return action, action_str

    def run_classifier_model(self):
        model_input = self.current_observation[np.newaxis, :]
        model_obs = model_input[:, :self.symptom_num * 3][:, 2::3]
        model_embedding = model_input[:, self.symptom_num * 3:]
        model_input = np.concatenate([model_obs, model_embedding], axis=1)
        result = self.classifier.predict_proba(model_input)

        diagnosis_dict = self.index_diagnosis_mapping
        diagnosis_list = []
        for i, item in enumerate(result[0]):
            diagnosis_list.append([item, i, diagnosis_dict[i]])

        diagnosis_list = sorted(diagnosis_list, key=lambda x: x[0], reverse=True)
        diagnosis_str = 'The final diagnosis list is:\n'
        for item in diagnosis_list:
            diagnosis_str += '#{}#\n'.format(item[1])
        return diagnosis_list, diagnosis_str

    def update_state(self, history):
        last_action = self.last_action
        question, response = history[-1]['question'], history[-1]['response']
        prompt = (
            "Please presume you are a doctor. You are in a diagnostic conversation. You have asked a question "
            "about the patient's symptoms. The question is:\n{}\n The patient replies:\n{}\n\nDoes the "
            "patient admit or deny the symptom?\n Respond with #YES# if they admit."
            "Response #NO# if they deny or say the requested information is not provided or is normal.\n"
            "You need to think carefully and response either #YES# or #NO#, but not both."
        ).format(question, response).strip()

        success_flag = False
        response = None
        failure_time = 0
        while not success_flag:
            try:
                if failure_time < 5:
                    response = call_llm(self.llm_name, prompt)
                    assert ((('#NO#' not in response) and '#YES#' in response) or
                            ('#NO#' in response and ('#YES#' not in response)))
                else:
                    response = '#NO#'
                assert self.current_observation[last_action * 3] == 1
                if '#YES#' in response:
                    self.current_observation[last_action * 3 + 2] = 1
                    self.current_observation[last_action * 3] = 0
                else:
                    self.current_observation[last_action * 3 + 1] = 1
                    self.current_observation[last_action * 3] = 0
                success_flag = True
            except Exception as exc:
                failure_time += 1
                logger.error('An error: {}, response: {}'.format(exc, response))

    def step(self, history):
        if len(history) > 0:
            self.update_state(history)

        if self.current_round < self.max_round:
            action, action_str = self.run_policy_model()
            prompt = (
                "Please presume you are a doctor. You are in a diagnostic conversation. "
                "You need to ask a question about the patient's symptoms. "
                "The question is \"{}\"."
                "Please respond only with a question.\n"
                "Note, you are asking only one question, not two questions.\n"
            ).format(action_str).strip()
            self.terminate = False
            response = call_llm(self.llm_name, prompt)
            self.last_action = action
        else:
            diagnosis_list, diagnosis_str = self.run_classifier_model()
            response = diagnosis_str
            self.terminate = True
        self.current_round += 1
        return response, self.terminate


class ScreenPureLLMDoctorSimulator(object):
    def __init__(self, data, max_round, llm_name, disease_range_dict):
        self.terminate = False
        self.conclusion = None
        self.current_round = 0

        self.data = data
        self.max_round = max_round
        self.disease_range_dict = disease_range_dict
        self.llm_name = llm_name
        self.current_observation = ['not_valid']
        self.disease_list_str = self.generate_disease_ranking_list()

    def reset(self):
        self.terminate = False
        self.conclusion = None
        self.current_round = 0

    def generate_disease_ranking_list(self):
        range_dict = self.disease_range_dict
        disease_list = []
        for group in range_dict:
            disease_list.append([len(disease_list), group])
        disease_list_str = "".join(["#{}# {}\n".format(item[0], item[1]) for item in disease_list])
        disease_list_str = disease_list_str.replace('\xa0', " ")
        return disease_list_str

    def step(self, history):
        if len(history) == 0:
            history_str = ''
        else:
            data_list = []
            for i, item in enumerate(history):
                question, response = item['question'], item['response']
                utt = 'ROUND: {}, DOCTOR ASK: {}, PATIENT RESPONSE: {}'.format(i, question, response)
                data_list.append(utt)
            history_str = '\n'.join(data_list)

        if self.current_round < self.max_round:
            prompt = (
                "Please presume you are a doctor. You are in a diagnosis conversation. You need to ask a series of "
                "questions (one question in one turn) to determine the disease of a patient. \n"
                "Previous dialogue history is:\n {}\n\n"
                "The previous medical history and surgical history is (if applicable) {}\n\n"
                "You are only allowed to ask patient present disease history during the conversation.\n"
                "Note, you are not allowed to ask patient lab test result or medical image exam result.\n"
                "This turn is the {} turn, the dialogue will end in next {} turns.\n"
                "When the dialogue is completed, you will need to give a high-risk disease list.\n"
                "Please ask questions as effective as possible.\n"
            ).format(history_str, self.data, self.current_round, self.max_round - self.current_round).strip()
            self.terminate = False
        else:
            prompt = (
                "Please presume you are a doctor. You are in a diagnosis conversation. You have complete the dialogue, "
                "and the dialogue history is:\n {}\n\n. Patients' previous medical history and surgical history is "
                "(if applicable) {}\n\nNow, based on this text, you are required to create a list of "
                "10 high-risk diseases that causes patient current admission. "
                "The diseases should be numbered in such a way that a smaller number indicates a "
                "higher risk associated with the disease.\n"
                "The disease candidate list is:\n{}\n."
                "The output format should be like:\n"
                "START:\n"
                "#3#\n"
                "#9#\n"
                "#4#\n"
                "#5#\n"
                "#12#\n"
                "#43#\n"
                "#28#\n"
                "#90#\n"
                "#16#\n"
                "#20#\n"
                "END\n"
                "if you think the disease #3# has the highest risk and #9# is the second highest risk disease, "
                "and so on.\n"
                "..."
            ).format(history_str, self.data, self.disease_list_str).strip()
            self.terminate = True
        response = call_llm(self.llm_name, prompt)
        self.current_round += 1
        return response, self.terminate


def read_data(path, folder_path, icd_code_group_dict, save_path, read_from_cache=True):
    if read_from_cache and os.path.exists(save_path):
        data_dict = pickle.load(open(save_path, 'rb'))
    else:
        emr_dict = dict()
        with (open(path, 'r', encoding='utf-8-sig', newline='') as f):
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                unified_id, patient_id, visit_id, note_id = line[0: 4]
                available_data_1 = ("Complaint: {}\n\nMajor Surgical or Invasive Procedure: {}\n\nHistory"
                                    " of Present Illness: {}\n\nPast Medical History: {}"
                                    ).format(line[7], line[8], line[9], line[10])
                available_data_2 = ("Major Surgical or Invasive Procedure: {} \n\nPast Medical History: {}"
                                    ).format(line[8], line[10])
                emr_dict[patient_id + '-' + visit_id] = available_data_1, available_data_2

        folder_list = os.listdir(folder_path)
        data_dict = dict()
        for sub_folder in folder_list:
            sub_folder_path = os.path.join(folder_path, sub_folder)
            file_set = set()
            for file_name in os.listdir(sub_folder_path):
                file_id = file_name.split('_')[0]
                file_set.add(file_id)
            for file_id in file_set:
                diagnosis_file_path = file_id + '_diagnosis.txt'
                emr_full_path = file_id + '_emr.txt'
                emr_key = '-'.join(file_id.strip().split('-')[1:])
                emr = emr_dict[emr_key]
                with open(os.path.join(folder_path, sub_folder, diagnosis_file_path), 'r', encoding='utf-8-sig') as f:
                    diagnosis = '\n'.join(f)
                    diagnosis = json.loads(diagnosis)
                with open(os.path.join(folder_path, sub_folder, emr_full_path), 'r', encoding='utf-8-sig') as f:
                    emr_full = '\n'.join(f.readlines())
                if '1' not in diagnosis:
                    logger.info('error {}'.format(file_id))
                    continue
                group = parse_disease_group(diagnosis['1']['code'], icd_code_group_dict)
                if group is None:
                    logger.info('group match failed {}'.format(file_id))
                    continue
                data_dict[file_id] = [emr[0], emr[1], diagnosis, group, emr_full]
        pickle.dump(data_dict, open(save_path, 'wb'))
    return data_dict


def parse_disease_group(source_icd, icd_group_dict):
    success_flag = False
    while not success_flag:
        if source_icd in icd_group_dict:
            group = icd_group_dict[source_icd]
            return group
        source_icd = source_icd[:-1]
    raise ValueError('')


def get_icd_mapping_code(path):
    icd_code_group_dict = dict()
    group_icd_code_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 3, None):
            icd_code, __, delete, english, chinese, _, group = line[0: 7]
            if group == '':
                assert delete == 'TRUE'
                continue
            if delete == 'TRUE':
                assert group == '' or group == '非疾病'
                continue
            icd_code_group_dict[icd_code] = group
            if group not in group_icd_code_dict:
                group_icd_code_dict[group] = list()
            group_icd_code_dict[group].append(icd_code)
    return group_icd_code_dict, icd_code_group_dict


if __name__ == '__main__':
    main()
