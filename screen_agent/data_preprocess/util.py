import csv
from itertools import islice
from screen_config import icd_9_mapping_file_path, diagnosis_icd_path
from logger import logger

def get_icd_mapping_dict():
    icd_dict = dict()
    with open(icd_9_mapping_file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd_9, icd_10 = line[0], line[1]
            icd_dict[icd_9] = icd_10
    return icd_dict


def get_mapped_diagnosis_icd(icd_dict):
    patient_diagnosis_dict = dict()
    success_count, error_count = 0, 0
    with open(diagnosis_icd_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id = line[0], line[1]
            if patient_id not in patient_diagnosis_dict:
                patient_diagnosis_dict[patient_id] = dict()
            if visit_id not in patient_diagnosis_dict[patient_id]:
                patient_diagnosis_dict[patient_id][visit_id] = dict()
            diagnosis_seq = line[2]
            try:
                if line[4] == "9":
                    icd_unified = icd_dict[line[3]]
                else:
                    assert line[4] == "10"
                    icd_unified = line[3]
                success_count += 1
                assert diagnosis_seq not in patient_diagnosis_dict[patient_id][visit_id]
                patient_diagnosis_dict[patient_id][visit_id][diagnosis_seq] = icd_unified
            except Exception as e:
                error_count += 1

    # 大约有千分之二的code无法正常mapping，所以直接忽略就可以
    logger.info('success count: {}, error count: {}'.format(success_count, error_count))
    return patient_diagnosis_dict