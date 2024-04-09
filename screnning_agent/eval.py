import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from logger import logger
from stable_baselines3.common.callbacks import BaseCallback
import copy
import torch
from datetime import datetime
import pickle


class DiagnosisEvalCallback(BaseCallback):
    def __init__(self, envs_train, envs_valid, envs_test, model, language, eval_per_step, episode_max_len,
                 save_path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_model = model
        self.eval_envs_train = envs_train
        self.eval_envs_valid = envs_valid
        self.eval_envs_test = envs_test
        self.eval_per_step = eval_per_step
        self.save_path = save_path
        self.language = language
        self.episode_max_len = episode_max_len

    def performance_eval(self, current_step, clf_iter_num=300):
        model = self.model
        max_step = self.eval_envs_train.envs[0].env.max_step

        result = []
        for eval_envs in (self.eval_envs_train, self.eval_envs_valid, self.eval_envs_test):
            obs_list = []
            symptom_list = []
            diagnosis_list = []
            with torch.no_grad():
                obs = eval_envs.reset()
                while not eval_envs.envs[0].unwrapped.epoch_end():
                    for env_monitor in eval_envs.envs:
                        diagnosis_list.append(copy.deepcopy(env_monitor.env.current_oracle_diagnosis))
                        symptom_list.append(copy.deepcopy(env_monitor.env.current_oracle_symptom))

                    for i in range(max_step):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = eval_envs.step(action)
                        if i < max_step - 1:
                            assert np.sum(dones) == 0
                        else:
                            assert np.sum(dones) == eval_envs.num_envs
                    # 这里因为自动重载，不能用final obs,并且要先注入diagnosis_list和symptom_list
                    for idx in range(len(eval_envs.buf_infos)):
                        obs_list.append(copy.deepcopy(eval_envs.buf_infos[idx]['terminal_observation']))
            # assert eval_envs.epoch_end()
            diagnosis = np.argmax(np.array(diagnosis_list), axis=1)
            observation = np.array(obs_list)
            symptom = np.array(symptom_list)
            result.append([diagnosis, observation, symptom])

        disease_num = self.eval_envs_train.envs[0].env.diagnosis_num
        first_level_action = self.eval_envs_train.envs[0].env.first_level_num
        clf = disease_predict_eval(result, disease_num, current_step, clf_iter_num, 2151)
        symptom_hit_eval(result, first_level_action, current_step, 2151)
        return clf

    def model_save(self, clf, current_step):
        model = self.model
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        policy_path = 'model_{}_{}_{}_{}_policy.pth'.format(self.language, self.episode_max_len, current_step, now)
        other_path = 'model_{}_{}_{}_{}_clf_dict.pkl'.format(self.language, self.episode_max_len, current_step, now)
        policy_path = os.path.join(self.save_path, policy_path)
        other_path = os.path.join(self.save_path, other_path)
        torch.save(model.policy.state_dict(), policy_path)
        pickle.dump(
            [
                clf,
                model.policy_kwargs['symptom_index_dict']
            ],
            open(other_path, 'wb')
        )

    def _on_training_start(self) -> None:
        clf = self.performance_eval(0, 0)
        self.model_save(clf, 0)

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        current_step = self.num_timesteps
        if current_step % self.eval_per_step != 0 and current_step != 0:
            return True
        clf = self.performance_eval(current_step)
        self.model_save(clf, current_step)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self.performance_eval(-1)
        pass


def symptom_hit_eval(result, first_level_action, step_num, obs_dim):
    train_symptom = np.array(result[0][2])
    test_symptom = np.array(result[2][2])
    observation_list_train = np.array(result[0][1])
    observation_list_test = np.array(result[2][1])

    train_observation = observation_list_train[:, :obs_dim]
    test_observation = observation_list_test[:, :obs_dim]
    train_match = train_observation * train_symptom
    test_match = test_symptom * test_observation

    symptom_num = len(test_match[0]) // 3
    train_first_level_hit, train_all_hit = 0, 0
    test_first_level_hit, test_all_hit = 0, 0
    train_symptom_first_level, test_symptom_first_level = 0, 0
    train_symptom_all, test_symptom_all = 0, 0
    for i in range(symptom_num):
        idx = i * 3 + 2
        if i < first_level_action:
            train_first_level_hit += np.sum(train_match[:, idx])
            test_first_level_hit += np.sum(test_match[:, idx])
            train_symptom_first_level += np.sum(train_symptom[:, idx])
            test_symptom_first_level += np.sum(test_symptom[:, idx])
        train_all_hit += np.sum(train_match[:, idx])
        test_all_hit += np.sum(test_match[:, idx])
        train_symptom_all += np.sum(train_symptom[:, idx])
        test_symptom_all += np.sum(test_symptom[:, idx])

    train_all_hit_average = train_all_hit / len(train_match)
    test_all_hit_average = test_all_hit / len(test_match)
    train_f_hit_average = train_first_level_hit / len(train_match)
    test_f_hit_average = test_first_level_hit / len(test_match)

    train_f_symptom_average = train_symptom_first_level / len(train_match)
    test_f_symptom_average = test_symptom_first_level / len(test_match)
    train_all_symptom_average = train_symptom_all / len(train_match)
    test_all_symptom_average = test_symptom_all / len(test_match)
    logger.info('step num: {:10d}, train avg sym: {:3.5f}, test avg sym: {:3.5f}, train first level sym: {:5f}, '
                'test first level sym: {:4f}'.format(step_num, train_all_symptom_average,
                                                     test_all_symptom_average, train_f_symptom_average,
                                                     test_f_symptom_average
    ))
    logger.info('step num: {:10d}, train avg hit: {:3.5f}, test avg hit: {:3.5f}, train first level hit: {:5f}, '
                'test first level hit: {:5f}'.format(step_num, train_all_hit_average, test_all_hit_average,
                                                     train_f_hit_average, test_f_hit_average
    ))


def disease_predict_eval(result, disease_num, epoch_num, clf_iter_num, obs_dim):
    diagnosis_list_train = np.array(result[0][0])
    observation_list_train = np.array(result[0][1])
    diagnosis_list_test = np.array(result[2][0])
    observation_list_test = np.array(result[2][1])

    train_obs_symptom = observation_list_train[:, :obs_dim][:, 2::3]
    train_obs_history = observation_list_train[:, obs_dim:]
    train_obs = np.concatenate([train_obs_symptom, train_obs_history], axis=1)
    test_obs_symptom = observation_list_test[:, :obs_dim][:, 2::3]
    test_obs_history = observation_list_test[:, obs_dim:]
    test_obs = np.concatenate([test_obs_symptom, test_obs_history], axis=1)

    max_iter = clf_iter_num
    clf = MLPClassifier(
        hidden_layer_sizes=[64, 32],
        # random_state=random_state,
        max_iter=max_iter,
        learning_rate_init=0.0001,
        learning_rate='adaptive',
        batch_size=len(train_obs) // 50,
    )
    for i in range(max_iter):
        clf.partial_fit(train_obs, diagnosis_list_train, classes=np.unique(diagnosis_list_train))
        if (i + 1) % 100 == 0 or i == max_iter - 1:
            train_diagnosis_prob = clf.predict_proba(train_obs)
            test_diagnosis_prob = clf.predict_proba(test_obs)
            for top_k in [1, 3, 5, 10, 20, 30, 50]:
                test_top_k_hit = top_calculate(diagnosis_list_test, test_diagnosis_prob, top_k, disease_num)
                train_top_k_hit = top_calculate(diagnosis_list_train, train_diagnosis_prob, top_k, disease_num)
                logger.info('step num: {:10d}, iter: {:4d}, train top {:2d} hit: {:5f},'
                            ' test top {:2d} hit: {:5f}'
                            .format(epoch_num, i + 1, top_k, train_top_k_hit, top_k, test_top_k_hit))
            logger.info('\n')
    return clf


def top_calculate(label, pred, top_k, length):
    data_size = len(label)

    top_k_hit_count = 0
    for i in range(data_size):
        sample_prob = np.zeros(length)
        sample_prob[:len(pred[i])] = pred[i]
        sample_label = np.zeros(len(sample_prob))
        sample_label[label[i]] = 1
        pair_list = []
        for j in range(len(sample_prob)):
            pair_list.append([sample_label[j], sample_prob[j]])

        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        for j in range(top_k):
            if pair_list[j][0] == 1:
                top_k_hit_count += 1
    top_k_hit = top_k_hit_count / data_size
    return top_k_hit
