import os
from datetime import datetime
from pathlib import Path
import torch

# hugginface_cache = os.path.abspath('/home/disk2/sunzhoujian/hugginface')
# root_folder = os.path.abspath('/home/disk2/sunzhoujian/screen_disease/resource')
root_path = Path(str(os.path.join(os.path.abspath(os.path.dirname(__file__))))).parent.absolute()
resource_folder = os.path.join(root_path, 'resource')
mimic_iv_folder = os.path.join(root_path, 'resource', 'mimic_iv')

prepared_data_pkl_path = os.path.join(mimic_iv_folder, 'prepared_data_{}_{}.pkl')
structured_data_folder = os.path.join(mimic_iv_folder, 'structured_symptom')
disease_reserve_path = os.path.join(mimic_iv_folder, 'disease_reserve_human_annotated.csv')
diagnosis_icd_path = os.path.join(mimic_iv_folder, 'diagnoses_icd.csv')
filtered_note_id_path = os.path.join(mimic_iv_folder, 'filtered_note_id.csv')
adult_symptom_path = os.path.join(mimic_iv_folder, 'symptom_adult.csv')
symptom_folder = os.path.join(mimic_iv_folder, 'structured_symptom')
transformed_id_list_path = os.path.join(mimic_iv_folder, 'transformed_id.csv')
reserved_note_path = os.path.join(mimic_iv_folder, 'reserved_note.csv')
disease_count_path = os.path.join(mimic_iv_folder, 'disease_count.csv')
embedding_transformed_id_list_file_path = os.path.join(mimic_iv_folder, 'embedding_transformed_id_list_file.csv')
discharge_path = os.path.join(mimic_iv_folder, 'discharge.csv')
d_icd_diagnoses = os.path.join(mimic_iv_folder, 'd_icd_diagnoses.csv')
history_folder = os.path.join(mimic_iv_folder, 'embedding')
# full admission embedding （包括主诉和现病史）
open_ai_full_admission_embedding_folder = os.path.join(root_path, 'resource', 'supplementary_experiment',
                                                       'open_ai_full_embedding')
# 不包括主诉和现病史
embedding_folder = os.path.join(mimic_iv_folder, 'open_ai_admission_embedding')
translation_path = os.path.join(mimic_iv_folder, 'translation.pkl')
data_fraction_path_template = os.path.join(mimic_iv_folder, 'data_fraction_id_list_{}_{}_{}_{}.pkl')
model_save_folder = os.path.join(resource_folder, 'screening_model_ckpt')
if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

default_model_name = 'ppo'
default_model_id = datetime.now().strftime('%Y%m%d%H%M%S')
default_first_level = 28
default_action_num = 717
default_learning_rate = 0.0001
default_value_weight = 0.5
default_entropy_weight = 0.01
default_n_envs = 256
default_episode_max_len = 10
default_update_per_step = 80
default_symptom_num = 717
default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
default_mode = 'collect'
default_use_text_embedding = True
default_value_net_length = 64

