import os
import json
from patient_simulator import PatientSimulator
from doctor_simulator import (DoctorSimulator, parse_questions, read_text)
from llm_util import call_open_ai
from environment import Environment
from datetime import datetime
from config import self_reflection_folder, diagnosis_procedure_template_path
import threading
from differential_util import evaluate_performance, GetNextBatch
from logger import logger


def run(dataset, positive_list, negative_list, batch_idx, doctor_llm_name, patient_llm_name, procedure_structure,
        start_index, disease, result_folder, flag_2):
    batch_data_dict = dict()
    for (flag_1, data_list) in (
            zip(('confirmed', 'excluded'), (positive_list, negative_list))):
        for key in data_list:
            data = dataset[key][0]
            path = os.path.join(result_folder, '{}_reflect_{}_{}_{}_{}_{}.json'
                                .format(disease, batch_idx, flag_1, key, doctor_llm_name, flag_2))
            batch_data_dict['-'.join([key, doctor_llm_name, flag_1, flag_2])] = [flag_1, key, data, path, flag_2]

    key_list = sorted(list(batch_data_dict.keys()))
    terminate = False
    id_batch = GetNextBatch(key_list, 20)
    while not terminate:
        threads = []
        batch, terminate = id_batch.next_batch()
        for key in batch:
            flag_1, key, data, path, flag_2 = batch_data_dict[key]

            # env patient和doctor都和线程绑定
            patient = PatientSimulator(patient_llm_name)
            ka_doctor = DoctorSimulator(procedure_structure, start_index, disease, doctor_llm_name)
            env = Environment(patient, ka_doctor)

            thread = threading.Thread(
                target=run_sample,
                args=(path, env, key, data))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


def run_sample(path, env, key, data):
    if not os.path.exists(path):
        result = env.run(key, data)
        result_str = json.dumps(result)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(result_str)




def generate_reflection(data, result_folder, batch_index, result_list, knowledge_text, knowledge_procedure, disease):
    reflection_list = list()
    for item in result_list:
        key, result = item[0], item[2]
        assert result in {'TP', "TN", "FP", "FN"}
        if result == 'TP' or result == 'TN':
            continue
        if result == 'FP':
            false_str = 'false positive'
        else:
            false_str = 'false negative'
        path_list = os.listdir(result_folder)
        valid_path_list = []
        for path in path_list:
            info_list = path.strip().split('_')
            batch = int(info_list[1])
            result_key = info_list[3]
            if batch == batch_index and key == result_key:
                valid_path_list.append(os.path.join(result_folder, path))
        assert len(valid_path_list) == 1
        json_str = json.load(open(valid_path_list[0], 'r', encoding='utf-8-sig'))
        dialogue = json_str['dialogue']
        dialogue_str = ''
        for i, turn in enumerate(dialogue):
            dialogue_str += ('Round: {}\nQuestion: {}\nAnswer:{}\n\n'
                             .format(i + 1, turn['question'], turn['response']))

        emr = data[key][0]
        prompt = (("Please assume that you are a senior physician. "
                   "You have summarized a structured {} diagnostic process (Attachment 1) from the clinical guidelines "
                   "(Attachment 2). When diagnosing a patient according to this diagnostic process, "
                   "a {} error occurred. The patient's EMR is shown as follows (Attachment 3), "
                   "and the conversation is shown as Attachment 4. Please think carefully and summarize "
                   "which question (response the number and question content) occurred in the "
                   "structured diagnosis procedure. Please answer briefly.\n\n "
                   "Attachment 1: {}\n\n Attachment 2: {}\n\n Attachment 3: {}\n\n Attachment 4: {}")
                  .format(disease, false_str, knowledge_procedure, knowledge_text, emr, dialogue_str))
        response = call_open_ai(prompt, 'gpt_4_turbo')
        reflection_list.append(response)
    return reflection_list



def revise_decision_procedure(disease, reflect_list, decision_procedure, decision_text, version_id, path_template):
    reflect_list_str = ''
    for i, reflect in enumerate(reflect_list):
        reflect_list_str += 'Reflect No. {}: {}\n\n\n\n'.format(i, reflect)
    prompt = (("Please assume you are an experienced cardiologist. You have summarized a structured "
               "diagnostic process (Attachment 1) for heart failure based on clinical guidelines (Attachment 2). "
               "Now, it has been found that this diagnostic process has encountered a series of issues in practice, "
               "with reflections found in Attachment 3. Please thoughtfully reconsider and generate a new, "
               "corrected structured diagnostic process. The format of the diagnostic process should adhere "
               "to the following format.\n "
               "(1) Please generate a decision procedure to describe the diagnosis of {} step by step "
               "based on the following clinical guideline information. Think carefully.\n "
               "(2) Each step of the procedure must be a question with \"Yes\" or \"No\" as the possible answers. "
               "Each answer must be mapped to a clear next step, such as jumping to another question or generating a "
               "conclusion and terminating the procedure.\n"
               "(3) If the answer maps to a jump step, the sentence should be like '#PROCEED TO QUESTION #n#', "
               "where all letters are in uppercase.\n"
               "(4) You need to summarize the title of the decision procedure by analyzing the context and the title.\n"
               "(5) The tree should be like: \n"
               "  Title: ...\n"
               "  Start \n"
               "  #QUESTION #1#: ... \n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  #QUESTION #2#: ... \n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  ...\n"
               "  #QUESTION #n#: ...\n"
               "      - Yes: ... \n"
               "      - No: ...\n"
               "  End\n"
               "(6) Please do not use information that is not included in the given context.\n"
               "(7) The final output of the decision procedure should explicitly express either "
               "\"YOU HAVE {}\" or \"YOU DON'T HAVE {}\". Other final outputs are not allowed.\n"
               "(8) Please make the procedure as precise as possible. It can be long, but it must not contain errors. "
               "(9) Questions are not allowed to have nested structure."
               "Each condition must have \"YES\" or \"NO\" as the two branches; no other branches are allowed.\n"
               "Attachment 1: {}\n\n Attachment 2: {}\n\n Attachment 3: {}")
              .format(disease, disease, disease, decision_procedure, decision_text, reflect_list_str))
    success_flag, start_index, response = False, None, None
    while not success_flag:
        response = call_open_ai(prompt, model_name='gpt_4_turbo')
        decision_procedure, start_index = parse_questions(response)
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(path_template.format(version_id, now), 'w', encoding='utf-8-sig', newline='') as f:
            f.writelines(response)
        success_flag = True
    return decision_procedure, start_index, response


def main():
    experiment_no = 'B'
    disease = "HEART FAILURE"
    patient_llm_name = 'gpt_4_turbo'
    doctor_llm_name = 'gpt_4_turbo'
    filter_name = 'gpt_4_turbo'
    full_positive_num = 160
    full_negative_num = 160
    batch_positive_num = 10
    batch_negative_num = 10
    train_portion = 0.375
    valid_portion = 0
    result_folder = os.path.join(self_reflection_folder, experiment_no)
    os.makedirs(result_folder, exist_ok=True)
    decision_procedure_path_dict = \
        {disease: diagnosis_procedure_template_path.format(2)}
    procedure_structure, start_index, procedure_text = read_text(decision_procedure_path_dict[disease])
    # with open(knowledge_origin_text_path, 'r', encoding='utf-8-sig') as f:
    #     diagnosis_text = '\n'.join(f.readlines())

    (dataset, batch_generate, valid_positive_list, valid_negative_list, test_positive_list, test_negative_list) = \
        get_data(disease, filter_name, full_positive_num, full_negative_num, train_portion, valid_portion,
                 batch_positive_num, batch_negative_num)

    logger.info('start analyze')

    target_idx = 2
    for i in range(6):
        batch_positive, batch_negative = batch_generate.get_next_batch()
        if i == target_idx:
            run(dataset, batch_positive, batch_negative, i, doctor_llm_name, patient_llm_name, procedure_structure,
                start_index, disease, result_folder, 'train')
            result_list, _ = evaluate_performance(result_folder)


if __name__ == '__main__':
    main()
