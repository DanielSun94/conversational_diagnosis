import copy
import random
import numpy as np
from numpy import int64
from gymnasium.spaces import Box, Discrete
from gymnasium import Env


class PatientEnvironment(Env):
    def __init__(self, data_pool, symptom_num, diagnosis_num, symptom_index_dict, first_level_symptom_num, max_step,
                 mode, random_sample, max_epoch, use_text_embedding, rank=None, total_env=None, embedding_size=None):
        super().__init__()
        self.data_pool = data_pool
        self.symptom_index_dict = symptom_index_dict
        self.symptom_num = symptom_num
        self.diagnosis_num = diagnosis_num
        self.first_level_num = first_level_symptom_num
        self.max_step = max_step
        self.mode = mode
        self.random_sample = random_sample
        self.max_epoch = max_epoch
        self.use_text_embedding = use_text_embedding
        self.rank = rank
        self.total_env = total_env
        self.embedding_size = embedding_size

        self.symptom_parent_idx_dict = {}
        for key in symptom_index_dict:
            idx, _, parent_idx = symptom_index_dict[key]
            self.symptom_parent_idx_dict[idx] = parent_idx
        # collect代表只询问症状，在询问完毕之后训练下一个模型进行诊断
        # diagnosis代表询问和诊断都纳入动作空间
        assert mode == 'collect' or mode == 'diagnosis'

        if mode == 'collect':
            self.action_space = Discrete(self.symptom_num)
        else:
            self.action_space = Discrete(self.symptom_num + self.diagnosis_num)
        if self.use_text_embedding:
            self.observation_space = Box(low=-10000, high=10000, shape=[self.symptom_num * 3 + 1 + embedding_size])
        else:
            self.observation_space = Box(low=-10000, high=10000, shape=[self.symptom_num * 3 + 1 + embedding_size])

        self.current_key = None
        self.current_oracle_symptom = None
        self.current_oracle_diagnosis = None
        self.current_oracle_embedding = None
        self.current_observation = None
        self.step_count = None

        self.iteration_list = None
        self.sample_idx = None
        self.epoch_num = 0
        # self.reset()

    def epoch_end(self):
        assert self.random_sample is False
        assert self.iteration_list is not None
        assert self.sample_idx <= len(self.iteration_list)
        if self.sample_idx == len(self.iteration_list):
            return True
        else:
            return False

    def step(self, action):
        if self.mode == 'collect':
            assert 0 <= action < self.symptom_num
        elif self.mode == 'diagnosis':
            assert 0 <= action < self.symptom_num + self.diagnosis_num
        else:
            raise ValueError('')
        assert self.current_oracle_diagnosis is not None and self.current_oracle_symptom is not None
        assert self.current_observation is not None
        assert isinstance(action, int64) or isinstance(action, int)
        if self.current_observation[-1] == 1:
            raise ValueError('')

        origin_observation = copy.deepcopy(self.current_observation)
        def _update_symptom_observation(origin_observation_, action_):
            current_unknown = origin_observation_[action_ * 3]
            new_observation_ = copy.deepcopy(origin_observation_)
            parent_idx = self.symptom_parent_idx_dict[action_]
            if parent_idx is not None:
                parent_unknown = origin_observation_[parent_idx * 3]
            else:
                parent_unknown = None
            no = self.current_oracle_symptom[action_ * 3 + 1]
            yes = self.current_oracle_symptom[action_ * 3 + 2]

            # 必须是之前没有问过的才能有positive reward
            if current_unknown == 1:
                if yes == 1:
                    new_observation_[action_ * 3] = 0
                    new_observation_[action_ * 3 + 2] = 1
                    new_observation_[action_ * 3 + 1] = 0
                    if action_ < 28:
                        reward_ = 1
                    else:
                        reward_ = 1
                else:
                    if parent_unknown is not None and parent_unknown == 1:
                        raise ValueError('')
                    elif no == 1:
                        reward_ = 0
                    else:
                        reward_ = 0
                    new_observation_[action_ * 3] = 0
                    new_observation_[action_ * 3 + 2] = 0
                    new_observation_[action_ * 3 + 1] = 1
            else:
                # 如果问了已经知道的信息，直接给负的reward
                assert current_unknown == 0
                # reward_ = -0.1
                reward_ = -0.1
            assert reward_ >=0
            return new_observation_, reward_

        if self.mode == 'diagnosis':
            # terminated是指新的状态是否是absorb state
            # 只有diagnosis会正常terminate（触发诊断即为terminate）
            if action < self.symptom_num:
                new_observation, reward = _update_symptom_observation(origin_observation, action)
                terminate = False
            else:
                diagnosis = self.current_oracle_diagnosis[action - self.symptom_num]
                if diagnosis == 1:
                    reward = 5
                else:
                    reward = -5
                new_observation = copy.deepcopy(origin_observation)
                new_observation[-1] = 1
                terminate = True
        else:
            new_observation, reward = \
                _update_symptom_observation(origin_observation, action)
            terminate = False

        # reward = reward - 0.1
        self.step_count += 1
        if self.step_count == self.max_step:
            truncate = True
        elif self.step_count < self.max_step:
            truncate = False
        elif self.step_count > self.max_step:
            raise ValueError('')
        else:
            raise ValueError('')
        self.current_observation = new_observation

        if self.use_text_embedding:
            return_obs = np.concatenate([new_observation, self.current_oracle_embedding])
        else:
            return_obs = np.concatenate([new_observation, np.zeros([self.embedding_size])])
        # if reward != 1 and reward!=0:
        #     raise ValueError('')
        return return_obs, reward, terminate, truncate, {}

    def reset(self, seed=None, options=None):
        if self.sample_idx is None or self.sample_idx == len(self.iteration_list):
            if self.random_sample:
                self.iteration_list = [i for i in range(len(self.data_pool))]
                random.Random().shuffle(self.iteration_list)
                self.sample_idx = 0
            else:
                assert isinstance(self.total_env, int) and isinstance(self.rank, int)
                start_idx = len(self.data_pool) // self.total_env * self.rank
                end_idx = len(self.data_pool) // self.total_env * (self.rank + 1)
                self.iteration_list = [i for i in range(start_idx, end_idx)]
                self.sample_idx = 0

        next_sample = self.data_pool[self.iteration_list[self.sample_idx]]
        self.current_key = next_sample[0]
        self.current_oracle_symptom = next_sample[1]
        self.current_oracle_diagnosis = next_sample[2]
        if self.use_text_embedding:
            self.current_oracle_embedding = next_sample[3]
        else:
            self.current_oracle_embedding = np.zeros([self.embedding_size])

        # 随机给定一个症状作为initial症状
        one_list = []
        for j in range(self.first_level_num):
            if self.current_oracle_symptom[j * 3 + 2] == 1:
                one_list.append(j * 3 + 2)

        next_observation = np.zeros([self.symptom_num * 3 + 1])
        next_observation[0::3] = 1
        next_observation[-1] = 0

        # 根据当前的设计（symptom > 1） one list长度一定大于1
        assert len(one_list) > 0
        if len(one_list) > 0:
            choice = int(random.choice(one_list))
            assert choice % 3 == 2
            next_observation[choice] = 1
            next_observation[choice // 3 * 3] = 0

        self.sample_idx += 1
        self.current_observation = next_observation
        self.step_count = 0
        self.epoch_num += 1
        assert next_observation[-1] == 0
        assert self.max_epoch is None or self.epoch_num <= self.max_epoch

        if self.use_text_embedding:
            embedding = np.array(self.current_oracle_embedding)
            return_obs = np.concatenate([next_observation, embedding])
        else:
            return_obs = np.concatenate([next_observation, np.zeros([self.embedding_size])])
        return return_obs, {}

    def render(self):
        print('call render, current key: {}'.format(self.current_key))

    def close(self):
        print('close, current key: {}'.format(self.current_key))


# def main():
#     first_level = 28
#     train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict = \
#         read_data_primary_diagnosis(structured_data_folder, minimum_symptom=1, read_from_cache=True)
#     symptom_dim = len(train_dataset.symptom[0]) // 3
#     diagnosis_dim = len(train_dataset.diagnosis[0])
#     mode = 'collect'
#     max_step = 20
#     _ = PatientEnvironment(
#         train_dataset, symptom_dim, diagnosis_dim, symptom_index_dict, first_level, max_step,
#         mode, True, True, None
#     )


#
# if __name__ == '__main__':
#     main()
