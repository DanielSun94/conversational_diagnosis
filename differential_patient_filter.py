import os.path
import csv
from llm_util import call_open_ai
import json
from config import logger, hf_differential_sample_path
import random
from patient_simulator_util import process_full_info_data


def heart_failure_confirm_by_icd(diagnosis):
    has_hf_flag = False
    for key in '1':
        if key not in diagnosis:
            continue
        code = diagnosis[key]['code']
        if code[0:3] == '428' or code[0:3] == 'I50':
            has_hf_flag = True
    return has_hf_flag


def heart_failure_exclude_by_icd(diagnosis):
    has_hf_flag = False
    for key in diagnosis:
        code = diagnosis[key]['code']
        if code in {'39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', "I9081",
                    'I110', 'I130', 'I132', 'I9713', 'I97130', 'I97131'}:
            has_hf_flag = True
        if code[0:3] == '428' or code[0:3] == 'I50':
            has_hf_flag = True
    return has_hf_flag


def heart_failure_confirm_check(info, key, llm_name):
    emr, diagnosis_str = info
    diagnosis = json.loads(diagnosis_str)
    has_hf_flag = heart_failure_confirm_by_icd(diagnosis)
    if not has_hf_flag:
        logger.info('key: {} excluded by icd confirm check'.format(key))
        return False

    prompt = (
        "Assuming you are a senior doctor specializing in cardiovascular diseases, please generate six answers on the "
        "six questions according to the given patient EMR. \n"
        "(1) Did the patient go to the hospital because of heart failure?\n"
        "If their first primary discharge diagnosis is heart failure, AND they have typical heart failure symptoms "
        "(such as breathlessness, orthopnoea, paroxysmal nocturnal dyspnoea, reduced exercise tolerance, fatigue,"
        "tiredness, increased time to recover after exercise, or ankle swelling) when they were admitted to "
        "the hospital, they can be identified as having gone to the hospital because of heart failure, vice versa.\n"
        "(2) Did the patient have a clear mind (not in a coma) during hospitalization?\n"
        "(3) Does the patient have a cardiac structural abnormality??\n"
        "(4) Did the patient undergo an echocardiogram and was the report recorded?\n"
        "(5) Was left ventricular ejection fraction recorded?\n"
        "(6) Did the EMR contain any electrocardiogram (ECG/EKG) information?\n"
        "NOTE:\n"
        "Please answer #YES# only when the answer to the question is yes and #NO# when the answer is no. "
        "(1) #YES#/#NO#\n"
        "(2) #YES#/#NO#\n"
        "(3) #YES#/#NO#\n"
        "(4) #YES#/#NO#\n"
        "(5) #YES#/#NO#\n"
        "(6) #YES#/#NO#\n"
        "PATIENT EMR: {}"
    ).format(info)

    success_flag = False
    flag_1, flag_2, flag_3, flag_4, flag_5, flag_6 = False, False, False, False, False, False
    while not success_flag:
        result = call_open_ai(prompt, llm_name)
        if not (("(1) #YES#" in result or "(1) #NO#" in result) and
                ("(2) #YES#" in result or "(2) #NO#" in result) and
                ("(3) #YES#" in result or "(3) #NO#" in result) and
                ("(4) #YES#" in result or "(4) #NO#" in result) and
                ("(5) #YES#" in result or "(5) #NO#" in result) and
                ("(6) #YES#" in result or "(6) #NO#" in result)):
            logger.info('{} parse invalid'.format(key))
            continue
        success_flag = True
        if "(1) #YES#" in result:
            flag_1 = True
        if "(2) #YES#" in result:
            flag_2 = True
        if "(3) #YES#" in result:
            flag_3 = True
        if "(4) #YES#" in result:
            flag_4 = True
        if "(5) #YES#" in result:
            flag_5 = True
        if "(6) #YES#" in result:
            flag_6 = True

    sufficiency = flag_1 and flag_2 and flag_3 and flag_4 and flag_5 and flag_6
    if not sufficiency:
        logger.info('{} excluded for heart failure confirm information insufficiency'.format(key))
    else:
        logger.info('{} included for positive list'.format(key))
    return sufficiency


def heart_failure_exclude_check(info, key, llm_name):
    emr, diagnosis_str = info
    diagnosis = json.loads(diagnosis_str)
    has_hf_flag = heart_failure_exclude_by_icd(diagnosis)
    if has_hf_flag:
        logger.info('key: {} excluded by icd exclude check'.format(key))
        return False

    prompt = (
        "Assuming you are a senior doctor specializing in cardiovascular diseases, please generate the answer on the "
        "the given question according to the given patient EMR. \n"
        "(1) Did the patient go to the hospital because of heart failure?\n"
        "If their PRIMARY discharge diagnoses include heart failure, or they have typical heart failure symptoms "
        "(such as breathlessness, orthopnoea, paroxysmal nocturnal dyspnoea, reduced exercise tolerance, fatigue,"
        "tiredness, increased time to recover after exercise, or ankle swelling) when they were admitted to "
        "the hospital, they can be identified as went to hospital because of heart failure, vice versa.\n"
        "NOTE:\n"
        "Please answer #YES# only when the answer to the question is yes and #NO# when the answer is no. "
        "Please answer using the format below:\n"
        "(1) #YES#/#NO# (brief reason)\n"
        "PATIENT EMR: {}"
    ).format(info)

    success_flag = False
    flag_1 = False
    while not success_flag:
        result = call_open_ai(prompt, llm_name)
        if not ("(1) #YES#" in result or "(1) #NO#" in result):
            logger.info('{} parse invalid'.format(key))
            continue
        success_flag = True
        if "(1) #NO#" in result:
            flag_1 = True

    if not flag_1:
        logger.info('{} excluded for heart failure exclude information insufficiency'.format(key))
    else:
        logger.info('{} included for negative list'.format(key))
    return flag_1


def write_pase_result(disease, positive_list, negative_list, disard_list):
    data_to_write = []
    for item in positive_list:
        data_to_write.append([disease, 'TRUE', item])
    for item in negative_list:
        data_to_write.append([disease, 'FALSE', item])
    for item in disard_list:
        data_to_write.append([disease, 'DISCARD', item])
    with open(hf_differential_sample_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def patient_filter(dataset, disease, positive_num=20, maximum_negative_num=200, llm_name='gpt_4_turbo',
                   read_from_cache=True):
    assert disease == 'hf'
    positive_list, negative_list, discard_list = list(), list(), list()
    if read_from_cache and os.path.exists(hf_differential_sample_path):
        with open(hf_differential_sample_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                disease, confirm, unified_id = line
                if confirm == "TRUE":
                    positive_list.append(unified_id)
                    pass
                elif confirm == "FALSE":
                    negative_list.append(unified_id)
                else:
                    assert confirm == 'DISCARD'
                    discard_list.append(unified_id)
    parsed_set = set(negative_list + positive_list+discard_list)
    index_list = [key for key in dataset]
    random.Random(715).shuffle(index_list)

    for key in index_list:
        if key in parsed_set:
            continue
        if len(positive_list) >= positive_num:
            break

        info = dataset[key]
        # 注意，confirm和exclude是互斥但不对立的。部分sample存在（根据指南）信息不足时完成诊断的情况
        # 因此技术上
        match_flag = False
        positive_flag = heart_failure_confirm_check(info, key, llm_name)
        if positive_flag:
            positive_list.append(key)
            match_flag = True
        if not positive_flag and len(negative_list) < maximum_negative_num:
            negative_flag = heart_failure_exclude_check(info, key, llm_name)
            if negative_flag:
                negative_list.append(key)
                match_flag = True
        # discard 有两种情况，一种是的确既不pos也不neg，另一种是negative太多被直接跳过
        # 由于本研究中的负样本远远多余正样本，因此我们允许部分negative被归入discard
        if not match_flag:
            discard_list.append(key)

        parsed_set.add(key)
        if len(positive_list+negative_list) % 10 == 0:
            write_pase_result(disease, positive_list, negative_list, discard_list)
    write_pase_result(disease, positive_list, negative_list, discard_list)
    logger.info('positive list: {}'.format(positive_list))
    logger.info('negative list: {}'.format(negative_list))
    return positive_list, negative_list


def main():
    positive_num = 160
    maximum_negative_num = 160
    disease = "hf"
    filter_name = 'gpt_4_turbo'
    patient_dataset = process_full_info_data()
    positive_list, negative_list = (
        patient_filter(patient_dataset, disease, positive_num, maximum_negative_num, filter_name))
    for item in positive_list:
        data = patient_dataset[item]
        print(data)
    for item in negative_list:
        data = patient_dataset[item]
        print(data)
    print('success')


if __name__ == '__main__':
    main()
