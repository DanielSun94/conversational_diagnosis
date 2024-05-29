import csv
from itertools import islice
from get_admission_emr import read_valid_emr_id
from screen_agent.screen_config import (unified_icd_path, unified_disease_count_path_template,
                                        discharge_path, d_icd_diagnoses)

discharge_emr_path = discharge_path
diagnosis_description_path = d_icd_diagnoses


def read_diagnosis_description(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd_code, icd_version, long_title = line
            data_dict[icd_code] = icd_version, long_title
    return data_dict


def read_mimic_iv_disease(top_k, key, valid_visit_set, code_num):
    assert key == 'circulatory system' or key == 'all'
    visit_id_set = set()
    valid_dict = dict()
    code_count, skip_visit_count = 0, 0
    with open(unified_icd_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, seq_num, icd_code = line
            unified_id = patient_id + '-' + visit_id
            icd_code = icd_code.lower()

            if unified_id not in valid_visit_set:
                skip_visit_count += 1
                continue

            seq_num = int(seq_num)
            if seq_num > top_k:
                continue

            visit_id_set.add(unified_id)
            if unified_id not in valid_dict:
                valid_dict[unified_id] = []

            code_count += 1
            icd_code = icd_code[:code_num]
            valid_dict[unified_id].append(icd_code)

    print('skip_visit_count: {}'.format(skip_visit_count))
    print('code_count: {}'.format(code_count))
    print('valid visit count: {}'.format(len(valid_dict)))
    print('single visit count: {}'.format(len(visit_id_set)))

    icd_count_dict = {}
    for unified_id in valid_dict:
        for icd_code in valid_dict[unified_id]:
            if icd_code not in icd_count_dict:
                icd_count_dict[icd_code] = 0
            icd_count_dict[icd_code] += 1
    icd_count_list = []
    for key in icd_count_dict:
        icd_count_list.append([key, icd_count_dict[key]])

    icd_count_list = sorted(icd_count_list, key=lambda x: x[1], reverse=True)
    return icd_count_list


# 1
def main():
    top_k, key, word_threshold = 100, 'all', 200
    code_number = 3
    valid_visit_set = read_valid_emr_id(discharge_emr_path, word_threshold)
    print('len valid set: {}'.format(len(valid_visit_set)))
    icd_count_list = read_mimic_iv_disease(top_k, key, valid_visit_set, code_number)
    print('load icd count list')
    diagnosis_dict = read_diagnosis_description(diagnosis_description_path)
    print('load diagnosis dict')
    disease_count_path = unified_disease_count_path_template.format(key, top_k, code_number)

    with open(disease_count_path, 'w', encoding='utf-8-sig', newline='') as f:
        data_to_write = [
            ['top k disease', top_k],
            ['key', key],
            ['icd_code', 'count', 'delete', 'English', 'Chinese']
        ]
        for line in icd_count_list:
            icd = line[0]
            if icd in diagnosis_dict:
                description = diagnosis_dict[icd][1]
            else:
                description = ''
            data_to_write.append([line[0], line[1], '', description])
        csv.writer(f).writerows(data_to_write)


if __name__ == '__main__':
    main()
