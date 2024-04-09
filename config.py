import os
import logging
import sys
from datetime import datetime

cache_folder = os.path.abspath('./resource/cache')
screen_folder = os.path.abspath('./resource/screen_result')
patient_info_group_path = os.path.abspath('./resource/patient_admission_info.pkl')
patient_info_folder = os.path.abspath('./resource/patient_admission_info')
self_reflection_folder = os.path.abspath('./resource/self_reflection')

reserve_note_file = os.path.abspath('./resource/mimic_iv/reserved_note.csv')
hf_differential_sample_path = os.path.abspath('./resource/heart_failure_differential_sample.csv')
code_mapping_path = os.path.abspath("./resource/disease_reserve_human_annotated.csv")
standard_question_path = os.path.abspath("./resource/standard_question.csv")

# screening only
planner_folder = os.path.abspath('./resource/screening_model_ckpt/')
embedding_folder = os.path.abspath('./resource/mimic_iv/open_ai_admission_embedding')
screening_planner_path_dict = {
    10: {
        'planner': os.path.join(planner_folder, 'model_english_10_3072000_20240328083748_policy.pth'),
        'classifier': os.path.join(planner_folder, 'model_english_10_3072000_20240328083748_clf_dict.pkl')
    },
    20: {
        'planner': os.path.join(planner_folder, 'model_english_20_2048000_20240328082753_policy.pth'),
        'classifier': os.path.join(planner_folder, 'model_english_20_2048000_20240328082753_clf_dict.pkl')
    }
}


# differential diagnosis only
patient_full_info_path = os.path.join('./resource/patient_full_info.pkl')
differential_folder = os.path.abspath('./resource/differential_diagnosis/')
differential_result_folder = os.path.join(differential_folder, 'differential_diagnosis')
knowledge_origin_text_path = os.path.join(differential_folder, 'heart_failure_diagnosis_content.txt')
reserve_key_path = os.path.join(differential_folder, 'transformed_id.csv')
diagnosis_procedure_init_path = os.path.join(differential_folder, 'heart_failure_decision_procedure_0.txt')
diagnosis_procedure_template_path = os.path.join(differential_folder, 'heart_failure_decision_procedure_{}.txt')

discharge_note_path = "F:\\MIMIC-IV\\note\\discharge.csv"
radiology_note_path = 'F:\\MIMIC-IV\\note\\radiology.csv'
diagnosis_path = 'F:\\MIMIC-IV\\hosp\\diagnoses_icd.csv'


log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource', 'log')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file_name = os.path.join(log_folder, '{}_{}.txt'.format('log', datetime.now().strftime("%Y%m%d%H%M%S'")))
format_ = "%(asctime)s %(process)d %(module)s %(lineno)d %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(log_file_name)
file_handler.setFormatter(logging.Formatter(format_))
logger.addHandler(file_handler)

# Create a stream handler for console output and set its level to INFO
console_logger = logging.StreamHandler(stream=sys.stdout)
console_logger.setLevel(logging.INFO)  # Setting the StreamHandler level to INFO
console_logger.setFormatter(logging.Formatter(format_))
logger.addHandler(console_logger)
