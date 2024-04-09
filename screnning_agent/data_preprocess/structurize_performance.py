import os
import csv
from itertools import islice

# skip
def main():
    folder_path = os.path.abspath('../resource/performance_check_sample')
    folder_list = os.listdir(folder_path)
    file_path_list = []
    for folder in folder_list:
        sub_folder_path = os.path.join(folder_path, folder)
        file_list = os.listdir(sub_folder_path)

        for file in file_list:
            if 'symptom_checked.csv' in file:
                file_path_list.append(os.path.join(sub_folder_path, file))

    assert len(file_path_list) == 80

    tp, tn, fp, fn = 0, 0, 0, 0
    for file in file_path_list:
        with open(file, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                symptom, factor_group, factor, state = line[0: 4]
                assert state == 'NA' or state == 'NO' or state == 'YES'
                if len(line) > 5:
                    line = line[:5]
                if len(line) == 4 or (len(line) == 5 and line[4] == ''):
                    if state == 'YES':
                        tp += 1
                    else:
                        tn += 1
                else:
                    if not (len(line) == 5 and line[4].lower() != ''):
                        raise ValueError('')
                    if state == 'YES':
                        fp += 1
                    else:
                        fn += 1

    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('ACC: {}'.format((tp + tn) / (tp + tn + fp + fn)))
    print('RECALL: {}'.format(tp / (tp + fn)))
    print('PRECISION: {}'.format(tp / (tp + fp)))


if __name__ == '__main__':
    main()
