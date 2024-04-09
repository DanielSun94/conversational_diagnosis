import csv
from itertools import islice
from util import call_open_ai
import os
import time
import threading
from datetime import datetime


def main():
    reserved_path = os.path.abspath('./resource/mimic_iv/reserved_note.csv')
    data_dict = dict()
    with open(reserved_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            note_id, complaint, history = line[3], line[4], line[6]
            data_dict[note_id] = 'Chief Complaint: \n' + complaint + '\n History of Present Illness: \n' + history

    performance_check_folder = os.path.abspath('./resource/performance_check_sample')
    key_list = list()
    for folder in os.listdir(performance_check_folder):
        folder_path = os.path.join(performance_check_folder, folder)
        for file in os.listdir(folder_path):
            if 'symptom.csv' not in file:
                continue
            unified_id = '-'.join(file.strip().split('_')[0].split('-')[:2])
            note_id = '-'.join(file.strip().split('_')[0].split('-')[2:])
            file_path = os.path.join(folder_path, unified_id + '-' + note_id + '_translated.csv')
            key_list.append([file_path, data_dict[note_id], False])

    success_num, batch_size = 0, 20
    while success_num < 400:
        batch = []
        for item in key_list:
            if item[2] is True:
                continue
            else:
                batch.append(item)
            if len(batch) >= 20:
                break

        results = [None] * len(batch)
        threads = []
        for i, item in enumerate(batch):
            time.sleep(0.05)
            thread = threading.Thread(
                target=translate,
                args=(item[1], results, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for item, result in zip(batch, results):
            if result is not None:
                item[2] = True
            data_to_write = [[result], [item[1]]]
            with open(item[0], 'w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerows(data_to_write)
        print('batch success')


def translate(context, results_list, index):
    prompt = "Please presume you are a senior doctor, please carefully translate the below content into Chinese." \
                 " Please carefully deal with the abbreviations:\n\n"
    prompt = prompt + context
    result = call_open_ai(prompt)
    # result = 'TBD'
    print(result)
    results_list[index] = result
    return


if __name__ == '__main__':
    main()
