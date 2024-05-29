import csv
from data_preprocess.util import get_icd_mapping_dict, get_mapped_diagnosis_icd
from screen_agent.screen_config import unified_icd_path


def main():
    icd_dict = get_icd_mapping_dict()
    patient_diagnosis_dict = get_mapped_diagnosis_icd(icd_dict)

    diagnosis_data = [['patient_id', 'visit_id', 'seq_num', 'icd_code']]
    data = []
    for patient_id in patient_diagnosis_dict:
        for visit_id in patient_diagnosis_dict[patient_id]:
            for seq_num in patient_diagnosis_dict[patient_id][visit_id]:
                icd_code = patient_diagnosis_dict[patient_id][visit_id][seq_num]
                data.append([patient_id, visit_id, int(seq_num), icd_code])
    data = diagnosis_data + data
    with open(unified_icd_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data)


if __name__ == '__main__':
    main()
