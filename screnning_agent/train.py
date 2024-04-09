from diagnosis_env import PatientEnvironment
from read_data import read_data
from config import structured_data_folder, model_save_folder, default_mode, default_device, default_model_id, \
    default_model_name, default_n_envs, default_action_num, default_symptom_num, default_first_level, \
    default_value_weight, default_entropy_weight, default_value_net_length, default_learning_rate,\
    default_episode_max_len, default_update_per_step, default_use_text_embedding
import argparse
from logger import logger
from stable_baselines3 import PPO, A2C, DQN
import torch as th
from util import make_vec_env, LinearSchedule
from eval import DiagnosisEvalCallback
from policy_model import SymptomInquireActorCriticPolicy


parser = argparse.ArgumentParser()
parser.add_argument('--language', help='', default='english', type=str)
parser.add_argument('--model_name', help='', default=default_model_name, type=str)
parser.add_argument('--model_id', help='', default=default_model_id, type=str)
parser.add_argument('--first_level', help='', default=default_first_level, type=int)
parser.add_argument('--action_num', help='', default=default_action_num, type=int)
parser.add_argument('--learning_rate', help='', default=default_learning_rate, type=float)
parser.add_argument('--value_weight', help='', default=default_value_weight, type=float)
parser.add_argument('--entropy_weight', help='', default=default_entropy_weight, type=float)
parser.add_argument('--n_envs', help='', default=default_n_envs, type=int)
parser.add_argument('--episode_max_len', help='', default=default_episode_max_len, type=int)
parser.add_argument('--update_per_step', help='', default=default_update_per_step, type=int)
parser.add_argument('--symptom_num', help='', default=default_symptom_num, type=int)
parser.add_argument('--device', help='', default=default_device, type=str)
parser.add_argument('--mode', help='', default=default_mode, type=str)
parser.add_argument('--use_text_embedding', help='', default=default_use_text_embedding , type=bool)
parser.add_argument('--value_net_length', help='', default=default_value_net_length , type=int)
args = vars(parser.parse_args())
for key in args:
    logger.info('{}: {}'.format(key, args[key]))


def main():
    n_envs = args['n_envs']
    init_learning_rate = args['learning_rate']
    first_level = args['first_level']
    episode_max_len = args['episode_max_len']
    mode = args['mode']
    device = args['device']
    update_per_step = args['update_per_step']
    use_text_embedding = args['use_text_embedding']
    value_weight = args['value_weight']
    entropy_weight = args['entropy_weight']
    value_net_length = args['value_net_length']
    language = args['language']
    assert language == 'chinese' or language == 'english'
    if use_text_embedding:
        embedding_size = 3072
    else:
        embedding_size = None
    train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict = \
        read_data(structured_data_folder, minimum_symptom=1, mode=language, read_from_cache=True)
    symptom_dim = len(train_dataset.symptom[0]) // 3
    diagnosis_dim = len(train_dataset.diagnosis[0])
    logger.info('data read success')

    envs_kwarg_train = {
        'first_level_symptom_num': first_level,
        'max_step': episode_max_len,
        'mode': mode,
        'data_pool': train_dataset,
        'symptom_num': symptom_dim,
        'diagnosis_num': diagnosis_dim,
        'use_text_embedding': use_text_embedding,
        'embedding_size': embedding_size,
        'symptom_index_dict': symptom_index_dict,
        'random_sample': True,
        'max_epoch': None,
    }

    envs_kwarg_eval_train = {
        'first_level_symptom_num': first_level,
        'max_step': episode_max_len,
        'mode': mode,
        'data_pool': train_dataset,
        'symptom_num': symptom_dim,
        'diagnosis_num': diagnosis_dim,
        'use_text_embedding': use_text_embedding,
        'embedding_size': embedding_size,
        'symptom_index_dict': symptom_index_dict,
        'random_sample': False,
        'max_epoch': None,
    }
    envs_kwarg_eval_valid = {
        'first_level_symptom_num': first_level,
        'max_step': episode_max_len,
        'mode': mode,
        'data_pool': valid_dataset,
        'symptom_num': symptom_dim,
        'diagnosis_num': diagnosis_dim,
        "use_text_embedding": use_text_embedding,
        'embedding_size': embedding_size,
        'symptom_index_dict': symptom_index_dict,
        'random_sample': False,
        'max_epoch': None,
    }
    envs_kwarg_eval_test = {
        'first_level_symptom_num': first_level,
        'max_step': episode_max_len,
        'mode': mode,
        'data_pool': test_dataset,
        'symptom_num': symptom_dim,
        'use_text_embedding': use_text_embedding,
        'embedding_size': embedding_size,
        'diagnosis_num': diagnosis_dim,
        'symptom_index_dict': symptom_index_dict,
        'random_sample': False,
        'max_epoch': None,
    }
    vec_env = make_vec_env(PatientEnvironment, n_envs=n_envs, env_kwargs=envs_kwarg_train)

    model_name = args['model_name']

    if model_name == 'dqn':
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 128, 128])
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_starts=n_envs * episode_max_len*10,
            buffer_size=819200,
            batch_size=n_envs * episode_max_len,  # envs * 10 epoch * 40 steps * 5 batch per epoch
            policy_kwargs=policy_kwargs,
            learning_rate=LinearSchedule(init_learning_rate),
            device=device,
            verbose=1
        )
    elif model_name == 'ppo':
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 128, 128], vf=[value_net_length]),
            symptom_index_dict=symptom_index_dict,
            symptom_num=symptom_dim
        )
        model = PPO(
            SymptomInquireActorCriticPolicy,
            vec_env,
            batch_size=n_envs*update_per_step,
            n_steps=update_per_step,
            policy_kwargs=policy_kwargs,
            learning_rate=LinearSchedule(init_learning_rate),
            ent_coef=entropy_weight,
            vf_coef=value_weight,
            device=device,
            verbose=1
        )
    elif model_name == 'a2c':
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 128, 128], vf=[256]),
            symptom_index_dict=symptom_index_dict,
            symptom_num=symptom_dim
        )
        model = A2C(
            SymptomInquireActorCriticPolicy,
            vec_env,
            n_steps=update_per_step,
            learning_rate=LinearSchedule(init_learning_rate),
            policy_kwargs=policy_kwargs,
            vf_coef=value_weight,
            ent_coef=entropy_weight,
            device=device,
            verbose=1
        )
    else:
        raise ValueError('')
    # 2048000约为10个epoch的结果 1024 env, 40 step, 5 batch per epoch, 10 epoch
    callback_interval = update_per_step * n_envs * 5 * 10
    vec_env_eval_train = make_vec_env(PatientEnvironment, n_envs=32, env_kwargs=envs_kwarg_eval_train)
    vec_env_eval_valid = make_vec_env(PatientEnvironment, n_envs=32, env_kwargs=envs_kwarg_eval_valid)
    vec_env_eval_test = make_vec_env(PatientEnvironment, n_envs=32, env_kwargs=envs_kwarg_eval_test)
    eval_callback = DiagnosisEvalCallback(
        vec_env_eval_train, vec_env_eval_valid, vec_env_eval_test, model, language, callback_interval, episode_max_len,
        model_save_folder)
    logger.info('start training')
    model.learn(
        total_timesteps=callback_interval * 6,
        log_interval=3,
        callback=eval_callback
    )
    logger.info('finish')


if __name__ == '__main__':
    main()
