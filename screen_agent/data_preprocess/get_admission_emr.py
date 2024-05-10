import csv
from itertools import islice
from screen_agent.screen_config import (
    reserved_note_path, disease_reserve_path, filtered_note_id_path, discharge_path, diagnosis_icd_path)


# 1
def main():
    reserved_icd_code_path = disease_reserve_path
    diagnosis_path = diagnosis_icd_path
    mimic4_file = discharge_path
    save_file = reserved_note_path
    filtered_file = filtered_note_id_path
    count_threshold, seq_num, word_threshold = 200, 1, 100
    token_list = ['Complaint:', 'Major Surgical or Invasive Procedure:', 'History of Present Illness:',
                  'Past Medical History:']

    valid_visit_diagnosis_dict, valid_diagnosis_dict = \
        read_valid_visit(diagnosis_path, reserved_icd_code_path, count_threshold, seq_num)
    valid_id_set = read_valid_emr_id(mimic4_file, word_threshold)
    print('len valid id set {}'.format(len(valid_id_set)))
    data_dict, filtered_list = read_emr(mimic4_file, token_list, valid_visit_diagnosis_dict)

    final_count = 0
    diagnosis_count_dict = dict()
    with open(save_file, 'w', encoding='utf-8-sig', newline='') as f:
        head = ['unified_id', 'patient_id', 'visit_id', 'note_id', 'diagnosis_group', 'diagnosis_description',
                'icd_code'] + token_list
        data_to_write = [head]
        for unified_id in data_dict:
            if unified_id not in valid_id_set:
                continue
            final_count += 1
            diagnosis_code = valid_visit_diagnosis_dict[unified_id]
            diagnosis_group = valid_diagnosis_dict[diagnosis_code][2]
            description = valid_diagnosis_dict[diagnosis_code][0]
            line = [unified_id, data_dict[unified_id]['patient_id'], data_dict[unified_id]['visit_id'],
                    data_dict[unified_id]['note_id'], diagnosis_group, description, diagnosis_code]

            if diagnosis_group not in diagnosis_count_dict:
                diagnosis_count_dict[diagnosis_group] = 0
            diagnosis_count_dict[diagnosis_group] += 1
            for key in token_list:
                line.append(data_dict[unified_id][key.lower()])
            data_to_write.append(line)
        csv.writer(f).writerows(data_to_write)
    print('final count: {}'.format(final_count))
    with open(filtered_file, 'w', encoding='utf-8-sig', newline='') as f:
        data_to_write = [['unified_id']]
        for item in filtered_list:
            data_to_write.append([item])
        csv.writer(f).writerows(data_to_write)

    diagnosis_count_list = []
    for key in diagnosis_count_dict:
        diagnosis_count_list.append([key, diagnosis_count_dict[key]])
    diagnosis_count_list = sorted(diagnosis_count_list, key=lambda x: x[1])
    print(diagnosis_count_list)


def read_valid_visit(diagnosis_path, reserved_icd_code_path, threshold, seq_num):
    valid_visit_diagnosis_dict = dict()
    valid_diagnosis_dict = dict()
    with open(reserved_icd_code_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 3, None):
            code, count, delete_flag, description, description_chinese = line[:5]
            disease_group = line[6]
            if len(code) == 0:
                continue
            if not count.isdigit():
                assert count.isdigit()
            assert len(code) <= 4
            if int(count) < threshold:
                continue
            if len(delete_flag) > 0:
                assert delete_flag == "TRUE"
                continue
            assert len(disease_group) > 0
            valid_diagnosis_dict[code.lower()] = description, description_chinese, disease_group

    with open(diagnosis_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, seq_num_str, icd_code, icd_version = line
            unified_id = patient_id + '-' + visit_id
            if int(seq_num_str) > seq_num:
                continue

            icd_code = icd_code.lower()[:4]
            if icd_code in valid_diagnosis_dict:
                valid_visit_diagnosis_dict[unified_id] = icd_code
    return valid_visit_diagnosis_dict, valid_diagnosis_dict


def read_emr(file_path, token_list, valid_visit_diagnosis_dict):
    data_dict = dict()
    filtered_list = list()

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            note_id, patient_id, visit_id, note_type, not_seq, chart_time, store_time, text = line
            unified_id = patient_id + '-' + visit_id
            if unified_id not in valid_visit_diagnosis_dict:
                continue

            lower_text = text.lower()
            valid_flag, text_dict = True, dict()
            for i in range(len(token_list)):
                key_1 = token_list[i].lower()
                position_1 = lower_text.find(key_1)
                if i == len(token_list) - 1:
                    position_2_1 = lower_text.find('Social History:'.lower())
                    position_2_2 = lower_text.find('Family History:'.lower())
                    if position_2_1 == -1 and position_2_2 == -1:
                        position_2 = -1
                    elif position_2_1 > 0 and position_2_2 > 0:
                        if position_2_1 > position_2_2:
                            position_2 = position_2_2
                        else:
                            position_2 = position_2_1
                    elif position_2_1 > 0:
                        position_2 = position_2_1
                    elif position_2_2 > 0:
                        position_2 = position_2_2
                    else:
                        raise ValueError('')
                else:
                    key_2 = token_list[i + 1].lower()
                    position_2 = lower_text.find(key_2)

                if position_1 == -1 or position_2 == -1:
                    valid_flag = False
                    filtered_list.append([note_id])
                    # print(unified_id)
                    break
                else:
                    text_dict[key_1] = text[position_1 + len(token_list[i]): position_2]
            if valid_flag:
                text_dict['patient_id'] = patient_id
                text_dict['visit_id'] = visit_id
                text_dict['note_id'] = note_id
                data_dict[unified_id] = text_dict

    word_count = 0
    for unified_id in data_dict:
        complaint, history = data_dict[unified_id]['complaint:'], data_dict[unified_id]['history of present illness:']
        word_count += len(complaint.split(' '))
        word_count += len(history.split(' '))
    print('average len: {}'.format(word_count/len(data_dict)))
    print('data size: {}'.format(len(data_dict)))
    print('filtered size: {}'.format(len(filtered_list)))
    return data_dict, filtered_list


def read_valid_emr_id(file_path, threshold):
    valid_visit_set = set()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id = line[1:3]
            emr = line[7].lower()
            position_1 = emr.find('History of Present Illness:'.lower())
            position_2 = emr.find('Past Medical History:'.lower())
            if position_1 <= 0 or position_2 <= 0:
                continue

            length = len(emr[position_1: position_2].split(' '))
            if length > threshold:
                valid_visit_set.add(patient_id+'-'+visit_id)
    return valid_visit_set


if __name__ == "__main__":
    main()
