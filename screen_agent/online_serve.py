import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import argparse
from screen_config import model_save_folder, adult_symptom_path
import numpy as np
from logger import logger
from openai import OpenAI
from fastapi import FastAPI, Body
import csv
import uvicorn
from requests.exceptions import HTTPError
from itertools import islice


model_file_name = 'model_10_4096000_20240229085547.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help='', default=model_file_name, type=str)
args = vars(parser.parse_args())
for arg in args:
    logger.info('{}: {}'.format(arg, args[arg]))


def get_model():
    llm_name = 'gpt-4-0125-preview'
    max_step = 10
    model_name = args['model_name']
    model_path = os.path.join(model_save_folder, model_name)
    symptom_dict = read_symptom(adult_symptom_path)
    policy_model, mlp_classifier, symptom_index_dict = pickle.load(open(model_path, 'rb'))

    model_runner = ModelRunner(policy_model, llm_name, symptom_index_dict, symptom_dict, max_step)
    return model_runner


def read_symptom(symptom_path):
    symptom_dict = dict()
    with open(symptom_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, factor_group, factor = line
            symptom, factor_group, factor = symptom.lower(), factor_group.lower(), factor.lower()
            if symptom not in symptom_dict:
                symptom_dict[symptom] = dict()
            if factor_group not in symptom_dict[symptom]:
                symptom_dict[symptom][factor_group] = []
            symptom_dict[symptom][factor_group].append(factor)
    return symptom_dict


class ModelRunner(object):
    def __init__(self, model_name, llm_name, symptom_name_dict, symptom_dict, max_step):
        self.model_name = model_name
        self.llm_name = llm_name
        index_symptom_dict = dict()
        self.symptom_name_dict = symptom_name_dict
        for symptom in symptom_name_dict:
            index_symptom_dict[symptom_name_dict[symptom][0]] = symptom
        self.state_meaning_dict = index_symptom_dict
        self.symptom_dict = symptom_dict
        self.max_step = max_step
        self.question_one, self.question_dict = self.construct_question_list()

        self.state = None
        self.dialogue_history = None
        self.current_step = None
        self.reset()

    def construct_question_list(self):
        symptom_dict = self.symptom_dict
        level_one_symptom_list = sorted(list(symptom_dict.keys()))
        prompt_1 = self.construct_level_one_question_list(level_one_symptom_list)
        prompt_2_dict = {}
        for key in symptom_dict:
            prompt_2_dict[key] = self.construct_level_two_question_list(key, symptom_dict[key])
        return prompt_1, prompt_2_dict

    @staticmethod
    def construct_level_two_question_list(symptom, factor_dict):
        prompt = 'Please answer whether the below factors exist when the give symptom: {} is existing. \n' \
                 'There are three possible answers for each factor. YES means a factor exists or ever exits based on ' \
                 'the given context, NO means a ' \
                 'factor does not exist, NA means a factor is not mentioned in the context\n' \
                 'PLEASE NOTE:\n' \
                 '1. a factor need to be treated as NA when it is not mentioned\n' \
                 '2. a factor need to be treated as exist (YES) when it directly relates or cause the current ' \
                 'hospital admission, if a factor exists but does not cause the current admission, ' \
                 'Please treats the symptom as NA\n ' \
                 '3. fever means patient body temperature larger or equal than 99 F\n'.format(symptom.upper())

        index = 0
        factor_group_list = sorted(list(factor_dict.keys()))
        for factor_group in factor_group_list:
            factor_list = sorted(factor_dict[factor_group])
            for item in factor_list:
                prompt += '#{}: {}, {}, {}\n'.format(index, symptom, factor_group, item)
                index += 1

        index = 0
        prompt += '\nPlease answer the question strictly according to the following format, without any other content\n'
        for factor_group in factor_group_list:
            factor_list = sorted(factor_dict[factor_group])
            for item in factor_list:
                prompt += '#{}: YES/NO/NA\n'.format(index, item)
                index += 1
        return prompt

    @staticmethod
    def construct_level_one_question_list(symptom_list):
        prompt = 'Please answer whether the below symptoms are existing. \n' \
                 'There are three possible answers for each symptom. YES means a symptom exists, NO means a ' \
                 'symptom does not exist, NA means a symptom is not mentioned in the context\n' \
                 'PLEASE NOTE:\n' \
                 '1. a factor need to be treated as NA when it is not mentioned\n' \
                 '2. a factor need to be treated as exist (YES) when it directly relates or cause the current ' \
                 'hospital admission, if a factor exists but does not cause the current admission, ' \
                 'Please treats the symptom as NA\n ' \
                 '3. fever means patient body temperature larger or equal than 99 F\n'
        for i, item in enumerate(symptom_list):
            prompt += '#{}#: {}\n'.format(i + 1, item)

        prompt += '\n Please answer the question strictly according to the following format, ' \
                  'without any other content\n'
        for i, item in enumerate(symptom_list):
            prompt += '#{}#, #{}#: YES/NO/NA\n'.format(i + 1, item)
        return prompt

    def step(self, user_response):
        if self.current_step == 0:
            next_utterance = self.initial_round_parse(user_response)
        else:
            binary_response = self.general_round_parse(user_response)
            self.update_state(binary_response)
            if self.current_step < self.max_step:
                next_action = self.generate_next_action()
                next_utterance = self.generate_next_question(next_action)
            else:
                diagnosis_list = self.implement_diagnosis()
                next_utterance = self.generate_diagnosis_response(diagnosis_list)
        self.current_step += 1
        return next_utterance

    def generate_diagnosis_response(self, diagnosis_list):
        return ''

    def implement_diagnosis(self):
        return ''

    def update_state(self, binary_response):
        pass

    def generate_next_action(self):
        pass

    def generate_next_question(self, next_action):
        return ''

    def general_round_parse(self, user_response):
        parse_flag = False
        response_state = None
        while not parse_flag:
            try:
                response_state = self.parse_symptom(user_response)
                parse_flag = True
            except HTTPError as http_err:
                # HTTP error occurred
                logger.error(f'HTTP error occurred: {http_err}')
            except Exception as err:
                # Other errors occurred
                logger.error(f'An error occurred: {err}')
            else:
                # Success
                logger.info('Success!')
        return response_state

    def initial_round_parse(self, user_input):
        parse_flag = False
        new_state = None
        while not parse_flag:
            try:
                new_state = self.parse_symptom(user_input)
                parse_flag = True
            except HTTPError as http_err:
                # HTTP error occurred
                logger.error(f'HTTP error occurred: {http_err}')
            except Exception as err:
                # Other errors occurred
                logger.error(f'An error occurred: {err}')
            else:
                # Success
                logger.info('Success!')
        return new_state

    def parse_symptom(self, user_input):
        symptom_dict = self.state_meaning_dict
        question_one = self.question_one
        question_dict = self.question_dict

        symptom_info, general_symptom_dict = self.initialize_symptom_info()
        init_prompt = "Please assume you are a senior doctor, given the below user utterance:\n\n"
        prompt = init_prompt + user_input + '\n' + question_one
        result = self.call_open_ai(prompt)

        result_list = result.strip().split('\n')
        symptom_list = list(symptom_dict.keys())
        assert len(result_list) == len(symptom_list)
        for result, symptom in zip(result_list, symptom_list):
            symptom = symptom.lower()
            assert ('NO' in result and 'NA' not in result and 'YES' not in result) or \
                   ('NO' not in result and 'NA' in result and 'YES' not in result) or \
                   ('NO' not in result and 'NA' not in result and 'YES' in result)

            if 'NO' in result and 'NA' not in result and 'YES' not in result:
                general_symptom_dict[symptom] = 'NO'
                for factor_group in symptom_info[symptom]:
                    for factor in symptom_info[symptom][factor_group]:
                        symptom_info[symptom][factor_group][factor] = 'NO'
            elif 'NO' not in result and 'NA' in result and 'YES' not in result:
                general_symptom_dict[symptom] = 'NA'
            elif 'NO' not in result and 'NA' not in result and 'YES' in result:
                general_symptom_dict[symptom] = 'YES'

                second_level_prompt = question_dict[symptom]
                self.parse_factor(user_input, symptom, symptom_info, second_level_prompt, init_prompt)
        return symptom_info

    def parse_factor(self, user_input, symptom, symptom_info, second_level_prompt, init_prompt):
        prompt = init_prompt + user_input + '\n' + second_level_prompt
        result = self.call_open_ai(prompt)

        result_list = result.strip().split('\n')
        factor_group_list = sorted(list(symptom_info[symptom]))
        factor_list = []
        for factor_group in factor_group_list:
            factors = sorted(list(symptom_info[symptom][factor_group].keys()))
            for factor in factors:
                factor_list.append([factor_group, factor])

        if len(result_list) != len(factor_list):
            assert len(result_list) == len(factor_list)
        for result, (factor_group, factor) in zip(result_list, factor_list):
            assert ('NO' in result and 'NA' not in result and 'YES' not in result) or \
                   ('NO' not in result and 'NA' in result and 'YES' not in result) or \
                   ('NO' not in result and 'NA' not in result and 'YES' in result)

            if 'NO' in result and 'NA' not in result and 'YES' not in result:
                symptom_info[symptom][factor_group][factor] = 'NO'
            elif 'NO' not in result and 'NA' in result and 'YES' not in result:
                symptom_info[symptom][factor_group][factor] = 'NA'
            elif 'NO' not in result and 'NA' not in result and 'YES' in result:
                symptom_info[symptom][factor_group][factor] = 'YES'

    def initialize_symptom_info(self):
        symptom_dict = self.symptom_dict
        symptom_info, general_symptom_dict = dict(), dict()
        for key in symptom_dict:
            symptom_info[key.lower()] = dict()
            general_symptom_dict[key.lower()] = "NA"
            for factor_group in symptom_dict[key]:
                symptom_info[key.lower()][factor_group.lower()] = dict()
                for factor in symptom_dict[key][factor_group]:
                    symptom_info[key.lower()][factor_group.lower()][factor.lower()] = "NA"
        return symptom_info, general_symptom_dict

    def reset(self):
        state = np.zeros(len(self.state_meaning_dict)*3)
        state[0::3] = 1
        self.state = state
        self.current_step = 0
        self.dialogue_history = []

        init_str = 'Nice to meet you. Where do you feel unwell?'
        self.dialogue_history.append(init_str)
        return init_str

    def call_open_ai(self, prompt):
        model_name = self.llm_name
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
        )
        response = chat_completion.choices[0].message.content
        return response

app = FastAPI()

model = get_model()


@app.post('/reset')
async def reset_session():
    system_utterance = model.reset()
    logger.info('model session reset')
    return system_utterance

@app.post('/step')
async def get_response(user_response: str = Body(..., embed=True)):
    model.step(user_response)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
