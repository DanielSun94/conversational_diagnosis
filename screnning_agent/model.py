import copy
import random
import numpy as np
import torch.nn.functional
from numpy import int64
from read_data import read_data
from config import structured_data_folder
from torch import nn
import torch.nn.functional as func
from torch import LongTensor
from torch.distributions import Categorical
import torch.nn.init as init


class ActorCritic(nn.Module):
    def __init__(self, symptom_num, action_num, hidden_sizes, gamma, e_weight, v_weight, device):
        assert symptom_num == 717 and (action_num == 717 or action_num == 815 or action_num == 28)
        super(ActorCritic, self).__init__()
        self.state_dim = symptom_num * 3 + 1
        self.action_num = action_num
        self.gamma = gamma
        self.device = device
        self.e_weight = e_weight
        self.v_weight = v_weight

        self.l1 = nn.Linear(self.state_dim, hidden_sizes[0]).to(device)
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(device)
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[1]).to(device)
        self.l4 = nn.Linear(self.state_dim, hidden_sizes[1]).to(device)
        self.l5 = nn.Linear(hidden_sizes[1], hidden_sizes[1]).to(device)
        self.l6 = nn.Linear(hidden_sizes[1], hidden_sizes[1]).to(device)
        self.actor = nn.Linear(hidden_sizes[1], action_num).to(device)
        self.critic = nn.Linear(hidden_sizes[1], 1).to(device)

        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

        # self.saved_actions = []
        # self.rewards = []
        # self.done = []
        # self.entropy = []

    def forward(self, state):
        x = func.tanh(self.l1(state))
        x = func.tanh(self.l2(x))
        x = func.tanh(self.l3(x))
        actor_logits = self.actor(x)

        y = func.tanh(self.l4(state))
        y = func.tanh(self.l5(y))
        y = func.tanh(self.l6(y))
        state_values = self.critic(y)  # Tensor of [bs, 1]
        return actor_logits, state_values

    @staticmethod
    def select_action(logits):
        # 此处默认随机选择True
        m = Categorical(logits=logits)
        probs = m.probs.detach().cpu().numpy()
        acts = m.sample()
        log_prob = m.log_prob(acts)
        return acts.cpu().numpy().tolist(), log_prob

    def update(self, optimizer, rollout_buffer):
        num_steps = len(rollout_buffer.action_list)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0

        action_list = rollout_buffer.action_list
        observation_list = rollout_buffer.origin_observation_list
        advantage_list = rollout_buffer.advantage_list
        valid_list = rollout_buffer.valid_list
        return_list = rollout_buffer.returns
        for i in range(0, num_steps):
            actions = LongTensor(np.array(action_list[i])).to(self.device)
            action_logits, value = self(observation_list[i].to(self.device))
            distribution = Categorical(logits=action_logits)
            log_prob = distribution.log_prob(actions)
            advantage = advantage_list[i]
            valid = valid_list[i]
            advantage = advantage * valid
            returns = return_list[i] * valid
            value = value.flatten()
            value = value * valid
            actor_loss += torch.mean(-log_prob * advantage)
            critic_loss += torch.nn.functional.mse_loss(returns, value)
            entropy = -distribution.entropy()
            entropy_loss += entropy.mean()
            # collect模式下应当符合全部valid

        actor_loss = actor_loss / num_steps
        critic_loss = critic_loss / num_steps
        entropy_loss = entropy_loss / num_steps
        # entropy_loss = entropy_loss.mean()
        loss = actor_loss + self.v_weight * critic_loss + self.e_weight * entropy_loss
        # loss = critic_loss
        # loss = actor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


class SinglePatientEnvironment(object):
    def __init__(self, symptom_num, diagnosis_num, symptom_index_dict,
                 data, first_level_symptom_num, terminate_step, mode):
        super().__init__()
        self.key = data[0]
        self.symptom_index_dict = symptom_index_dict
        self.oracle_symptom = data[1]
        self.oracle_diagnosis = data[2]
        self.symptom_num = symptom_num
        self.diagnosis_num = diagnosis_num
        self.first_level_num = first_level_symptom_num
        self.terminate_step = terminate_step
        self.step_count = None
        self.observation = None
        self.mode = mode

        self.parent_idx_dict = {}
        for key in symptom_index_dict:
            idx, _, parent_idx = symptom_index_dict[key]
            self.parent_idx_dict[idx] = parent_idx
        assert mode == 'collect' or mode == 'diagnosis'

    def step(self, action):
        if self.mode == 'collect':
            assert 0 <= action < self.symptom_num
        elif self.mode == 'diagnosis':
            assert 0 <= action < self.symptom_num + self.diagnosis_num
        else:
            raise ValueError('')
        assert isinstance(action, int64) or isinstance(action, int)
        assert self.mode == 'collect'

        valid_step = True
        origin_observation = copy.deepcopy(self.observation)
        if self.observation[-1] == 1:
            raise ValueError('')
            # valid_step = False
            # return origin_observation, origin_observation, 0, True, valid_step

        def _update_symptom_observation(origin_observation_, action_):
            current_unknown = origin_observation_[action_ * 3]
            new_observation_ = copy.deepcopy(origin_observation_)
            parent_idx = self.parent_idx_dict[action_]
            if parent_idx is not None:
                parent_unknown = origin_observation_[parent_idx * 3]
            else:
                parent_unknown = None
            yes = self.oracle_symptom[action_ * 3 + 2]
            no = self.oracle_symptom[action_ * 3 + 1]

            # 必须是之前没有问过的才能
            if current_unknown == 1:
                if yes == 1:
                    new_observation_[action_ * 3 + 2] = 1
                    new_observation_[action_ * 3 + 1] = 0
                    new_observation_[action_ * 3] = 0
                    # if action_ < 28:
                    #     reward_ = 1
                    # else:
                    #     reward_ = 0.3
                    reward_ = 1
                else:
                    if parent_unknown is not None and parent_unknown == 1:
                        reward_ = -0.1
                    elif no == 1:
                        reward_ = 0
                    else:
                        reward_ = 0
                    new_observation_[action_ * 3 + 2] = 0
                    new_observation_[action_ * 3 + 1] = 1
                    new_observation_[action_ * 3] = 0
            else:
                # 如果问了已经知道的信息，直接给负的reward
                assert current_unknown == 0
                # reward_ = -0.1
                reward_ = -0.1
            return new_observation_, reward_

        if self.mode == 'diagnosis':
            if action < self.symptom_num:
                new_observation, reward = \
                    _update_symptom_observation(origin_observation, action)
            else:
                diagnosis_idx = action - self.symptom_num
                diagnosis = self.oracle_diagnosis[diagnosis_idx]
                if diagnosis == 1:
                    reward = 5
                else:
                    reward = -5
                new_observation = copy.deepcopy(origin_observation)
                new_observation[-1] = 1

            self.step_count += 1
            if self.step_count > self.terminate_step:
                raise ValueError('')
            # if self.step_count == self.terminate_step:
            #     if reward != 5:
            #         reward = -5
            # 正常衰减
            reward = reward - 0.1
            if new_observation == 1:
                terminated = True
            else:
                terminated = False
            # terminated是指新的状态是否是absorb state

            self.observation = new_observation
            return origin_observation, new_observation, reward, terminated, valid_step
        else:
            new_observation, reward = \
                _update_symptom_observation(origin_observation, action)
            self.observation = new_observation
            # reward = reward - 0.1
            return origin_observation, new_observation, reward, False, True

    def reset(self):
        # 随机给定一个症状作为initial症状
        current_observation = np.zeros([self.symptom_num * 3 + 1])
        for i in range(self.symptom_num):
            current_observation[i * 3] = 1
        one_list = []
        for j in range(self.first_level_num):
            if self.oracle_symptom[j * 3 + 2] == 1:
                one_list.append(j * 3 + 2)

        # 根据当前的设计（symptom > 5） one list长度一定大于1
        assert len(one_list) > 0
        if len(one_list) > 0:
            choice = int(random.choice(one_list))
            assert choice % 3 == 2
            current_observation[choice] = 1
            current_observation[choice // 3 * 3] = 0
        self.observation = current_observation
        self.step_count = 0
        assert current_observation[-1] == 0
        return current_observation


class DiagnosisEnvironment(object):
    def __init__(self, symptom_num, diagnosis_num, dataset, first_level_num, batch_size, terminate_step,
                 diagnosis_index_dict, symptom_index_dict, mode, seed=None):
        self.symptom_num = symptom_num
        self.diagnosis_num = diagnosis_num
        self.dataset = dataset
        self.first_level_num = first_level_num
        self.batch_size = batch_size
        self.data_size = len(dataset.symptom)
        self.terminate_step = terminate_step
        self.diagnosis_index_dict = diagnosis_index_dict
        self.symptom_index_dict = symptom_index_dict
        self.mode = mode
        self.seed = seed
        self.random_index_list = [i for i in range(self.data_size)]
        assert batch_size < self.data_size

        self.current_batch_index = 0
        self.current_batch = None
        self.current_step = None
        self.reset()

    def reset(self):
        if self.seed is not None:
            random.Random(self.seed).shuffle(self.random_index_list)
        else:
            random.Random().shuffle(self.random_index_list)
        self.current_batch_index = 0

    def get_next_batch(self):
        if (self.current_batch_index + 1) * self.batch_size >= self.data_size:
            raise ValueError('')

        start_idx = self.current_batch_index * self.batch_size
        end_idx = (self.current_batch_index + 1) * self.batch_size
        next_batch_idx_list = self.random_index_list[start_idx: end_idx]
        next_batch = []

        observations = []
        for index in next_batch_idx_list:
            key, symptom, diagnosis = self.dataset.__getitem__(index)
            single_patient = SinglePatientEnvironment(
                self.symptom_num, self.diagnosis_num, self.symptom_index_dict,
                (key, symptom, diagnosis), self.first_level_num,
                self.terminate_step, self.mode
            )
            observation = single_patient.reset()
            observations.append(observation)
            next_batch.append(single_patient)
        self.current_batch_index += 1
        self.current_batch = next_batch
        self.current_step = 0
        return observations

    def has_next(self):
        if (self.current_batch_index + 1) * self.batch_size < self.data_size:
            return True
        else:
            return False

    def step(self, actions):
        if self.current_step >= self.terminate_step:
            raise ValueError('')
        result = []
        for (action, env) in zip(actions, self.current_batch):
            original_obs, new_obs, reward, terminated, valid = env.step(action)
            result.append([original_obs, new_obs, reward, terminated, valid])
        self.current_step += 1
        return result


def main():
    batch_size = 32
    first_level = 28
    train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict = \
        read_data(structured_data_folder, minimum_symptom=1, read_from_cache=True)
    symptom_dim = len(train_dataset.symptom[0]) // 3
    diagnosis_dim = len(train_dataset.diagnosis[0])
    mode = 'collect'
    terminate_step = 20
    env = DiagnosisEnvironment(symptom_dim, diagnosis_dim, train_dataset, first_level, batch_size, terminate_step,
                               diagnosis_index_dict, symptom_index_dict, mode)
    for i in range(1000):
        env.get_next_batch()
    for i in range(20):
        env.step([j * 25 + i for j in range(batch_size)])
    env.get_next_batch()
    for i in range(20):
        env.step([j * 25 + i for j in range(batch_size)])


if __name__ == '__main__':
    main()
