import csv
import os
import time
import random
from itertools import islice
import threading
from datetime import datetime

THREAD_NUM = 100
MAX_NUM = 10000
FOLDER_SIZE = 100

model = 'gpt-4-turbo'


def main():
    full_emr_path = os.path.abspath('./resource/mimic_iv/reserved_note.csv')
    symptom_path = os.path.abspath('./resource/symptom_adult.csv')
    symptom_folder = os.path.abspath('./resource/mimic_iv/structured_symptom')
    transformed_id_list_path = os.path.abspath('./resource/transformed_id.csv')
    symptom_dict = read_symptom(symptom_path)
    id_seq_list = read_id_order_seq(full_emr_path, random_seed=715)
    question_one, question_dict = construct_question_list(symptom_dict)
    data_dict = read_context(full_emr_path)

    if os.path.exists(transformed_id_list_path):
        transformed_id_dict = dict()
        with open(transformed_id_list_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                transformed_id_dict[line[0]] = line[1]
    else:
        transformed_id_dict = dict()

    count_id = len(transformed_id_dict)
    while count_id < MAX_NUM:
        start_time = datetime.now()
        unified_ids = get_next_batch_id(data_dict, transformed_id_dict, id_seq_list, THREAD_NUM)
        if count_id >= MAX_NUM:
            break

        results = [[None, None, None, None]] * len(unified_ids)
        threads = []
        for i, unified_id in enumerate(unified_ids):
            time.sleep(0.05)
            thread = threading.Thread(
                target=parse_symptom,
                args=(data_dict, unified_id, symptom_dict, question_one, question_dict, results, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for result in results:
            if result[0] is None or result[1] is None or result[2] is None or result[3] is None:
                continue
            general_symptom_dict, symptom_info, dialogue_list, unified_id = result
            formatted_num = "{:05d}".format(len(transformed_id_dict) // FOLDER_SIZE)
            folder_path = os.path.join(symptom_folder, formatted_num)
            os.makedirs(folder_path, exist_ok=True)

            symptom_path = os.path.join(folder_path, unified_id + '_symptom.csv')
            with open(symptom_path, 'w', encoding='utf-8-sig', newline='') as f:
                data_to_write = [['symptom', 'factor_group', 'factor', 'state']]
                for symptom in general_symptom_dict:
                    data_to_write.append([symptom, 'N/A', 'N/A', general_symptom_dict[symptom]])
                for symptom in symptom_info:
                    for factor_group in symptom_info[symptom]:
                        for factor in symptom_info[symptom][factor_group]:
                            state = symptom_info[symptom][factor_group][factor]
                            data_to_write.append([symptom, factor_group, factor, state])
                csv.writer(f).writerows(data_to_write)

            detail_path = os.path.join(folder_path, unified_id + '_detail.csv')
            with open(detail_path, 'w', encoding='utf-8-sig', newline='') as f:
                data_to_write = [['question', 'answer']]
                for item in dialogue_list:
                    data_to_write.append(item)
                csv.writer(f).writerows(data_to_write)

            transformed_id_dict[unified_id] = formatted_num
            with open(transformed_id_list_path, 'w', encoding='utf-8-sig', newline='') as f:
                data_to_write = [['unified_id', 'folder_num']]
                for unified_id in transformed_id_dict:
                    data_to_write.append([unified_id, transformed_id_dict[unified_id]])
                csv.writer(f).writerows(data_to_write)
            print('unified_id: {} success'.format(unified_id))
            count_id += 1

        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Time difference: {time_diff}")


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


def get_next_batch_id(data_dict, transformed_id_dict, id_seq_list, batch_size):
    id_list = []
    for key in id_seq_list:
        if key in data_dict and key not in transformed_id_dict:
            id_list.append(key)
        if len(id_list) >= batch_size:
            break
    return id_list


def parse_symptom(data_dict, unified_id, symptom_dict, question_one, question_dict, results, index):
    dialogue_list = []
    content = data_dict[unified_id]

    symptom_info, general_symptom_dict = initialize_symptom_info(symptom_dict)
    init_prompt = "Please assume you are a senior doctor, given the below admission record:\n\n"
    prompt = init_prompt + content + '\n' + question_one
    result = call_open_ai(prompt, model_name=model)

    dialogue_list.append([prompt, result])

    result_list = result.strip().split('\n')
    symptom_list = list(symptom_dict.keys())
    assert len(result_list) == len(symptom_list)
    for result, symptom in zip(result_list, symptom_list):
        symptom = symptom.lower()
        assert ('NO' in result and 'NA' not in result and 'YES' not in result) or \
               ('NO' not in result and 'NA' in result and 'YES' not in result) or \
               ('NO' not in result and 'NA' not in result and 'YES' in result)

        if 'NO' in result and 'NA' not in result and 'YES' not in result:
            general_symptom_dict[symptom] = 'NO'
            for factor_group in symptom_info[symptom]:
                for factor in symptom_info[symptom][factor_group]:
                    symptom_info[symptom][factor_group][factor] = 'NO'
        elif 'NO' not in result and 'NA' in result and 'YES' not in result:
            general_symptom_dict[symptom] = 'NA'
        elif 'NO' not in result and 'NA' not in result and 'YES' in result:
            general_symptom_dict[symptom] = 'YES'

            second_level_prompt = question_dict[symptom]
            parse_factor(content, symptom, symptom_info, second_level_prompt, init_prompt, dialogue_list)

    results[index] = general_symptom_dict, symptom_info, dialogue_list, unified_id
    return


def parse_factor(content, symptom, symptom_info, second_level_prompt, init_prompt, dialogue_list):
    prompt = init_prompt + content + '\n' + second_level_prompt
    result = call_open_ai(prompt, model_name=model)

    dialogue_list.append([prompt, result, symptom])
    result_list = result.strip().split('\n')

    factor_group_list = sorted(list(symptom_info[symptom]))
    factor_list = []
    for factor_group in factor_group_list:
        factors = sorted(list(symptom_info[symptom][factor_group].keys()))
        for factor in factors:
            factor_list.append([factor_group, factor])

    if len(result_list) != len(factor_list):
        assert len(result_list) == len(factor_list)
    for result, (factor_group, factor) in zip(result_list, factor_list):
        assert ('NO' in result and 'NA' not in result and 'YES' not in result) or \
               ('NO' not in result and 'NA' in result and 'YES' not in result) or \
               ('NO' not in result and 'NA' not in result and 'YES' in result)

        if 'NO' in result and 'NA' not in result and 'YES' not in result:
            symptom_info[symptom][factor_group][factor] = 'NO'
        elif 'NO' not in result and 'NA' in result and 'YES' not in result:
            symptom_info[symptom][factor_group][factor] = 'NA'
        elif 'NO' not in result and 'NA' not in result and 'YES' in result:
            symptom_info[symptom][factor_group][factor] = 'YES'
    return


def initialize_symptom_info(symptom_dict):
    symptom_info, general_symptom_dict = dict(), dict()
    for key in symptom_dict:
        symptom_info[key.lower()] = dict()
        general_symptom_dict[key.lower()] = "NA"
        for factor_group in symptom_dict[key]:
            symptom_info[key.lower()][factor_group.lower()] = dict()
            for factor in symptom_dict[key][factor_group]:
                symptom_info[key.lower()][factor_group.lower()][factor.lower()] = "NA"
    return symptom_info, general_symptom_dict


def read_symptom(symptom_path):
    symptom_dict = dict()
    with open(symptom_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, factor_group, factor = line
            symptom, factor_group, factor = symptom.lower(), factor_group.lower(), factor.lower()
            if symptom not in symptom_dict:
                symptom_dict[symptom] = dict()
            if factor_group not in symptom_dict[symptom]:
                symptom_dict[symptom][factor_group] = []
            symptom_dict[symptom][factor_group].append(factor)
    return symptom_dict


def construct_question_list(symptom_dict):
    level_one_symptom_list = sorted(list(symptom_dict.keys()))
    prompt_1 = construct_level_one_question_list(level_one_symptom_list)
    prompt_2_dict = {}
    for key in symptom_dict:
        prompt_2_dict[key] = construct_level_two_question_list(key, symptom_dict[key])
    return prompt_1, prompt_2_dict


def construct_level_two_question_list(symptom, factor_dict):
    prompt = 'Please answer whether the below factors exist when the give symptom: {} is existing. \n' \
             'There are three possible answers for each factor. YES means a factor exists or ever exits based on ' \
             'the given context, NO means a ' \
             'factor does not exist, NA means a factor is not mentioned in the context\n' \
             'PLEASE NOTE:\n' \
             '1. "deny" or "denies" a symptom means NO. \n' \
             '2. a factor need to be treated as NA when it is not mentioned\n' \
             '3. a factor need to be treated as exist (YES) when it directly relates or cause the current ' \
             'hospital admission, if a factor exists but does not cause the current admission, ' \
             'Please treats the symptom as NA\n ' \
             '4. fever means patient body temperature larger or equal than 99 F\n'.format(symptom.upper())

    index = 0
    factor_group_list = sorted(list(factor_dict.keys()))
    for factor_group in factor_group_list:
        factor_list = sorted(factor_dict[factor_group])
        for item in factor_list:
            index += 1
            prompt += '#{}: {}, {}, {}\n'.format(index, symptom, factor_group, item)

    index = 0
    prompt += '\nPlease answer the question strictly according to the following format, without any other content\n'
    for factor_group in factor_group_list:
        factor_list = sorted(factor_dict[factor_group])
        for item in factor_list:
            index += 1
            prompt += '#{}: YES/NO/NA\n'.format(index, item)
    # print(prompt)
    return prompt


def construct_level_one_question_list(symptom_list):
    prompt = 'Please answer whether the below symptoms are existing. \n'\
             'There are three possible answers for each symptom. YES means a symptom exists, NO means a ' \
             'symptom does not exist, NA means a symptom is not mentioned in the context\n'\
             'PLEASE NOTE:\n' \
             '1. "deny" or "denies" a symptom means NO. \n' \
             '2. a factor need to be treated as NA when it is not mentioned\n' \
             '3. a factor need to be treated as exist (YES) when it directly relates or cause the current ' \
             'hospital admission, if a factor exists but does not cause the current admission, ' \
             'Please treats the symptom as NA\n ' \
             '4. fever means patient body temperature larger or equal than 99 F\n'
    for i, item in enumerate(symptom_list):
        prompt += '#{}#: {}\n'.format(i+1, item)

    prompt += '\n Please answer the question strictly according to the following format, without any other content\n'
    for i, item in enumerate(symptom_list):
        prompt += '#{}#, #{}#: YES/NO/NA\n'.format(i+1, item)
    # print(prompt)
    return prompt


def read_context(discharge_path):
    data_dict = dict()
    with open(discharge_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            unified_id, note_id = line[0], line[3]
            complaint, present_illness = line[7], line[9]
            context = 'Chief Complaint:\n {} \n\n\nHistory of Present Illness: {}\n'.format(complaint, present_illness)
            data_dict[unified_id+'-'+note_id] = context
    return data_dict


if __name__ == '__main__':
    main()
