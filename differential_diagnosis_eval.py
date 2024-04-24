import os
import json
import random
import threading
from patient_simulator import PatientSimulator
from differential_util import evaluate_performance, get_data, GetNextBatch
from doctor_simulator import (PureGPTDoctorSimulator, TextKnowledgeGPTDoctorSimulator, DoctorSimulator, read_text)
from environment import Environment
from config import differential_result_folder, knowledge_origin_text_path, diagnosis_procedure_template_path
from logger import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disease', help='', default='hf', type=str)
parser.add_argument('--doctor_llm_name', help='gpt_4_turbo, llama3-70b',
                    default='llama3-70b', type=str)
parser.add_argument('--model_type', help='text_knowledge_gpt, ka_gpt, pure_gpt',
                    default='text_knowledge_gpt', type=str)
parser.add_argument('--eval_mode', help='', default=1, type=int)
parser.add_argument('--version_index', help='', default=0, type=int)
parser.add_argument('--thread_num', help='', default=1, type=int)
args = vars(parser.parse_args())
for arg in args:
    logger.info('{}: {}'.format(arg, args[arg]))


def main():
    disease = args['disease']
    doctor_llm_name = args['doctor_llm_name']
    model_type = args['model_type']
    eval_mode = args['eval_mode']
    version_index = args['version_index']
    thread_num = args['thread_num']
    assert eval_mode == 0 or eval_mode == 1
    eval_mode = True if eval_mode == 1 else False
    assert isinstance(version_index, int)

    full_positive_num = 160
    full_negative_num = 160
    train_portion = 0.375
    valid_portion = 0
    patient_dataset, _, _, _, test_positive_list, test_negative_list = (
        get_data(disease, 'gpt_4_turbo', full_positive_num, full_negative_num, train_portion, valid_portion,
                 1, 1))

    if not eval_mode:
        # sample_num = None
        os.makedirs(differential_result_folder, exist_ok=True)
        with open(knowledge_origin_text_path, 'r', encoding='utf-8-sig') as f:
            diagnosis_text = '\n'.join(f.readlines())

        decision_procedure_path_dict = \
            {disease: diagnosis_procedure_template_path.format(version_index)}
        procedure_structure, start_index, procedure_text = read_text(decision_procedure_path_dict[disease])

        logger.info('start analyze')
        setting_dict = dict()
        for flag_1, id_list in zip(('confirmed', 'excluded'), (test_positive_list, test_negative_list)):
            for unified_id in id_list:
                patient = PatientSimulator('gpt_4_turbo')
                setting_key = str('-'.join([flag_1, unified_id, model_type]))
                if model_type == 'pure_gpt':
                    assert version_index == 0
                    doctor = PureGPTDoctorSimulator(disease, doctor_llm_name)
                    env = Environment(patient, doctor)
                    setting_dict[setting_key] = [flag_1, unified_id, model_type, env]
                elif model_type == 'text_knowledge_gpt':
                    assert version_index == 0
                    doctor = TextKnowledgeGPTDoctorSimulator(disease, diagnosis_text, doctor_llm_name)
                    env = Environment(patient, doctor)
                    setting_dict[setting_key] = [flag_1, unified_id, model_type, env]
                else:
                    assert model_type == 'ka_gpt'
                    doctor = DoctorSimulator(procedure_structure, start_index, disease, doctor_llm_name)
                    env = Environment(patient, doctor)
                    setting_dict[setting_key] = [flag_1, unified_id, model_type, env]

        setting_list = list(setting_dict.keys())
        random.Random().shuffle(setting_list)

        id_batch = GetNextBatch(setting_list, thread_num)
        terminate = False
        while not terminate:
            threads = []
            batch, terminate = id_batch.next_batch()
            for key in batch:
                flag_1, unified_id, flag_2, env = setting_dict[key]
                data = patient_dataset[unified_id][0]
                path = os.path.join(
                    differential_result_folder, '{}_{}_{}_{}_{}_{}_test.json'
                    .format(disease, version_index, flag_1, unified_id, doctor_llm_name, flag_2))
                if not os.path.exists(path):
                    thread = threading.Thread(
                        target=run_sample,
                        args=(path, key, data, env, flag_1, flag_2, doctor_llm_name))
                    threads.append(thread)
                    thread.start()
                else:
                    logger.info('{} existed'.format(unified_id))
            for thread in threads:
                thread.join()
    logger.info('test data')
    evaluate_performance(differential_result_folder, test_positive_list + test_negative_list)
    logger.info('including previous result')
    evaluate_performance(differential_result_folder, None)


def run_sample(path, key, data, env, flag_1, flag_2, doctor_llm_name):
    result = env.run(key, data)
    result_str = json.dumps(result)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(result_str)
    logger.info('success {}, {}, {}, {}'.format(key, flag_1, flag_2, doctor_llm_name))


if __name__ == '__main__':
    main()
