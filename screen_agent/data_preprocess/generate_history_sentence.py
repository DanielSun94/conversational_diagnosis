import csv
import os
from llm_util import call_open_ai
from logger import logger
import threading
from datetime import datetime
import re
import json
from screen_agent.screen_config import (discharge_path, open_ai_full_admission_embedding_folder,
                                        fixed_question_answer_folder)

THREAD_NUM = 200
MAX_NUM = 40000
FOLDER_SIZE_NUM = 100

prompt_template = (
    'Please play the role of a patient and answer the following six questions based on the given '
    'electronic medical record. Three questions need to be answered in English, and three questions need '
    'to be answered in Chinese. Please answer use natural sentences, like a real human.\n'
    'English Questions:\n'
    'Question 1: Please provide your gender, age, height, and weight information. If this information is not '
    'recorded in the electronic medical record, please say "I don\'t know."\n'
    'Question 2: Please tell me what illnesses you have had before. If this information is not '
    'recorded in the electronic medical record, please say "I don\'t know."\n'
    'Question 3: What illnesses have your family members had before? If this information is not '
    'recorded in the electronic medical record, please say "I don\'t know."\n'
    'Chinese Questions:\n'
    '问题1：请告诉我你的性别，年龄，身高，体重信息。如果电子病例中未记录相关信息，请回答我不知道。\n'
    '问题2：请告诉我你以前得过什么病？如果电子病例中未记录相关信息，请回答我不知道。\n'
    '问题3：请问你家里人之前得过什么病？如果电子病例中未记录相关信息，请回答我不知道。\n\n'
    '请按照如下格式回复\n'
    'English Question Answer:\n'
    '#1#: (answer)\n'
    '#2#: (answer)\n'
    '#3#: (answer)\n'
    'Chinese Question Answer:\n'
    '#1#: (answer)\n'
    '#2#: (answer)\n'
    '#3#: (answer)\n\n'
    'The electronic medical record is given as below:\n{}'
)


def read_valid_id_set():
    valid_id_set = set()
    folder_list = os.listdir(open_ai_full_admission_embedding_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(open_ai_full_admission_embedding_folder, folder))
        for file in file_list:
            valid_id_set.add(file[:-4])
    logger.info('valid id set size: {}'.format(len(valid_id_set)))
    return valid_id_set


def read_next_batch_info(batch_size, valid_id_set, processed_id_set):
    next_batch_data = []
    with open(discharge_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            note_id, subject_id, hadm_id, note_type, note_seq, chartt_ime, store_time, text = line
            unified_id = subject_id + "-" + hadm_id
            if unified_id not in processed_id_set and unified_id in valid_id_set:
                next_batch_data.append([unified_id, text])
            if len(next_batch_data) >= batch_size:
                break
    return next_batch_data


def generate_fixed_question_answer(emr_text, unified_id, processed_id_set, index, reserved_length=600):
    """reserved length 600是因为之前算过既往史的平均长度约为300词，增加两倍应该通常够了"""
    folder_num = index // FOLDER_SIZE_NUM
    folder = os.path.join(fixed_question_answer_folder, str(folder_num))
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, unified_id + ".json")
    if os.path.exists(file_path):
        processed_id_set.add(unified_id)
        logger.info('{} already exists'.format(file_path))
    else:
        failure_time = 0
        success_flag = False
        while not success_flag:
            try:
                if failure_time > 4:
                    break
                reserved_text = ' '.join(emr_text.split(' ')[:reserved_length])
                prompt = prompt_template.format(reserved_text)
                response = call_open_ai(prompt, 'gpt_35_turbo')
                parsed_response = parse_response(response)
                json.dump(parsed_response, open(file_path, 'w', encoding='utf-8-sig'))
                processed_id_set.add(unified_id)
                success_flag = True
            except Exception as e:
                logger.info(e)
                logger.info('{} failed'.format(unified_id))
                failure_time += 1


def parse_response(response):
    result = {'en': {}, 'zh': {}}
    assert 'English Question Answer:' in response
    assert 'Chinese Question Answer:' in response
    chinese_start_idx = response.find('Chinese Question Answer:')
    matches = re.findall(r"#(\d+)#: (.+?)(?=#\d+|$)", response[:chinese_start_idx], re.DOTALL)
    assert len(matches) == 3
    for match in matches:
        number, content = match
        if number == '1':
            result['en']['basic_information'] = content
        elif number == '2':
            result['en']['past_medical_history'] = content
        else:
            assert number == '3'
            result['en']['family_history'] = content

    matches = re.findall(r"#(\d+)#: (.+?)(?=#\d+|$)", response[chinese_start_idx:], re.DOTALL)
    assert len(matches) == 3
    for match in matches:
        number, content = match
        if number == '1':
            result['zh']['基本信息'] = content
        elif number == '2':
            result['zh']['既往史'] = content
        else:
            assert number == '3'
            result['zh']['家族史'] = content
    return result


def reload_processed_id_set():
    processed_id_set = set()
    folder_list = os.listdir(fixed_question_answer_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(fixed_question_answer_folder, folder))
        for file in file_list:
            file_name = file[:-5]
            processed_id_set.add(file_name)
    return processed_id_set


# 用于工程化部署所需的既往史，家族史的encoding
def main():
    valid_id_set = read_valid_id_set()
    assert len(valid_id_set) == 40000
    processed_id_set = reload_processed_id_set()

    while len(processed_id_set) < MAX_NUM:
        start_time = datetime.now()
        batch_data = read_next_batch_info(THREAD_NUM, valid_id_set, processed_id_set)

        threads = []
        preprocessed_data_len = len(processed_id_set)
        for i, (unified_id, emr_text) in enumerate(batch_data):
            thread = threading.Thread(
                target=generate_fixed_question_answer,
                args=(emr_text, unified_id, processed_id_set, i + preprocessed_data_len))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = datetime.now()
        time_diff = end_time - start_time
        logger.info("Time difference: {}, processed_id_set length: {}".format(time_diff, len(processed_id_set)))
    logger.info('finished generating fixed question answer')


if __name__ == '__main__':
    main()
