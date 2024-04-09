import csv
from itertools import islice
from get_admission_emr import read_valid_emr_id
from screnning_agent.config import diagnosis_icd_path, disease_count_path, discharge_path, d_icd_diagnoses

mimic4_file_path = diagnosis_icd_path
disease_count_path = disease_count_path
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


def read_mimic_iv_disease(top_k, key, valid_visit_set, code_num=4):
    assert key == 'circulatory system' or key == 'all'
    visit_id_set = set()
    valid_dict = dict()
    with open(mimic4_file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, seq_num, icd_code, icd_version = line
            unified_id = patient_id + "-" + visit_id
            visit_id_set.add(unified_id)
            icd_code = icd_code.lower()

            if unified_id not in valid_visit_set:
                continue

            seq_num = int(seq_num)
            if seq_num > top_k:
                continue

            assert icd_version == '9' or icd_version == '10'
            if key == 'circulatory system':
                if icd_version == '9' and icd_code.isdigit():
                    icd_code = int(icd_code[:code_num])
                    if 390 <= icd_code <= 459:
                        valid_dict[unified_id] = icd_code
                else:
                    if icd_code[0].lower() == 'i':
                        valid_dict[unified_id] = icd_code
            else:
                assert key == 'all'
                if icd_version == '9' and icd_code.isdigit():
                    icd_code = int(icd_code[:code_num])
                    valid_dict[unified_id] = str(icd_code)
                valid_dict[unified_id] = str(icd_code)[:code_num]
    print('single visit count: {}'.format(len(visit_id_set)))
    print('valid visit count: {}'.format(len(valid_dict)))

    icd_count_dict = {}
    for unified_id in valid_dict:
        if valid_dict[unified_id] not in icd_count_dict:
            icd_count_dict[valid_dict[unified_id]] = 0
        icd_count_dict[valid_dict[unified_id]] += 1
    icd_count_list = []
    for key in icd_count_dict:
        icd_count_list.append([key, icd_count_dict[key]])

    icd_count_list = sorted(icd_count_list, key=lambda x: x[1], reverse=True)
    return icd_count_list


# 1
def main():
    top_k, key, threshold = 1, 'all', 100
    valid_visit_set = read_valid_emr_id(discharge_emr_path, threshold)
    print('len valid set: {}'.format(len(valid_visit_set)))
    icd_count_list = read_mimic_iv_disease(top_k, key, valid_visit_set)
    diagnosis_dict = read_diagnosis_description(diagnosis_description_path)

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
