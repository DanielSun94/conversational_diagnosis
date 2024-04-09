import re
from llm_util import call_llm


def read_text(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        procedure_text = "".join(f.readlines())
    decision_procedure, start_index = parse_questions(text=procedure_text)
    return decision_procedure, start_index, procedure_text


def parse_questions(text):
    question_dict = dict()
    start_index = text.find('Start') + 5
    text = text[start_index:]
    pattern = r"#QUESTION\s+#\d+#"
    # Using findall to find all occurrences in the text
    matches = re.findall(pattern, text)
    for i, _ in enumerate(matches):
        start_index = text.find(matches[i]) + len(matches[i])
        if i < len(matches) - 1:
            end_index = text.find(matches[i + 1])
        else:
            end_ = text[start_index:]
            if 'End' in end_:
                end_index = end_.find('End') + start_index
            else:
                end_index = len(text)
        condition_str = text[start_index: end_index]
        question_content = parse_condition(condition_str)

        question = matches[i].strip().split('#')
        question_index = int(question[-2])
        question_dict[question_index] = question_content
    return question_dict, int(matches[0].strip().split('#')[-2])


def parse_condition(text):
    assert "- Yes:" in text and "- No:" in text
    condition = text[: text.find('- Yes:')]
    yes = text[text.find('- Yes:') + len("- Yes:"): text.find('- No:')]
    no = text[text.find('- No:') + len("- No:"):]

    # nested question error
    assert '- Yes:' not in yes and '- No:' not in yes
    assert '- Yes:' not in no and '- No:' not in no
    condition_obj = Condition(condition, yes, no)
    return condition_obj


class Condition(object):
    def __init__(self, condition, yes, no):
        self.condition = condition
        self.yes = self.parse(yes)
        self.no = self.parse(no)

    def yes_next(self):
        return self.yes

    def no_next(self):
        return self.no

    @staticmethod
    def parse(text):
        assert 'YOU HAVE' in text or "YOU DON'T HAVE" in text or 'PROCEED' in text
        if "YOU HAVE" in text:
            return text, 'DIAGNOSIS CONFIRM'
        elif "YOU DON'T HAVE" in text:
            return text, "DIAGNOSIS EXCLUDE"
        else:
            assert 'PROCEED' in text
            question = text.strip().split('#')
            question_index = int(question[-2])
            return question_index, 'NAVIGATE'


class TextKnowledgeGPTDoctorSimulator(object):
    def __init__(self, disease, knowledge, llm_name):
        self.disease = disease
        self.terminate = False
        self.knowledge = knowledge
        self.conclusion = None
        self.llm_name = llm_name

        if self.disease == 'hf':
            self.disease_full_name = 'heart failure'
        else:
            assert ValueError('')

    def reset(self):
        self.terminate = False
        self.conclusion = None

    def step(self, history):
        if len(history) == 0:
            history_str = ''
        else:
            data_list = []
            for i, item in enumerate(history):
                question = item['question']
                response = item['response']
                utt = 'ROUND: {}, DOCTOR ASK: {}, PATIENT RESPONSE: {}'.format(i, question, response)
                data_list.append(utt)
            history_str = '\n'.join(data_list)
        prompt = (("Please assume you are a doctor specializing in cardiovascular diseases. "
                   "You are engaged in a {} diagnosis conversation. Your task is to ask a series of questions "
                   "(one question per turn) to ascertain whether the patient has {}. The previous dialogue "
                   "history is as follows:\n {}\n\n "
                   "Note: You may inquire about any patient information, including laboratory test results "
                   "and medical examination findings.\n"
                   "You should return #CONFIRM# if you believe the patient has the disease, "
                   "or #EXCLUDE# if you believe the patient does not have the disease.\n"
                   "Otherwise, please formulate the next question you need to ask based on the "
                   "following diagnostic knowledge:\n {}\n\n\n")
                  .format(self.disease_full_name, self.disease_full_name, history_str, self.knowledge))

        response = call_llm(self.llm_name, prompt)
        if "#CONFIRM#" in response:
            terminate = True
            response = 'DIAGNOSIS CONFIRM'
        elif "#EXCLUDE#" in response:
            terminate = True
            response = 'DIAGNOSIS EXCLUDE'
        elif len(history) > 20:
            terminate = True
            response = 'TOO LONG FAILED'
        else:
            terminate = False
        self.terminate = terminate
        return response, terminate


class PureGPTDoctorSimulator(object):
    def __init__(self, disease, llm_name):
        self.disease = disease
        self.terminate = False
        self.conclusion = None
        self.llm_name = llm_name

        if self.disease == 'hf':
            self.disease_full_name = 'heart failure'
        else:
            assert ValueError('')

    def reset(self):
        self.terminate = False
        self.conclusion = None

    def step(self, history):
        if len(history) == 0:
            history_str = ''
        else:
            data_list = []
            for i, item in enumerate(history):
                question = item['question']
                response = item['response']
                utt = 'ROUND: {}, DOCTOR ASK: {}, PATIENT RESPONSE: {}'.format(i, question, response)
                data_list.append(utt)
            history_str = '\n'.join(data_list)

        prompt = (('Please assume you are a doctor specializing in cardiovascular disease. '
                   'You are engaged in a {} diagnosis conversation.\n'
                   'You need to ask a series of questions (one question at a time) to determine whether the '
                   'patient has {}. The previous dialogue history is:\n {}\n\n'
                   'You can ask for any information about the patient, including lab test results and '
                   'medical exam outcomes. If you think the information is sufficient, '
                   'please confirm or exclude the diagnosis. You need to return #CONFIRM# if you believe '
                   'the patient has the disease or #EXCLUDE# if you think the patient does not have the disease.\n'
                   'Otherwise, please generate the next question you need to ask.')
                  .format(self.disease_full_name, self.disease_full_name, history_str))

        response = call_llm(self.llm_name, prompt)
        if "#CONFIRM#" in response:
            terminate = True
            response = 'DIAGNOSIS CONFIRM'
        elif "#EXCLUDE#" in response:
            terminate = True
            response = 'DIAGNOSIS EXCLUDE'
        elif len(history) > 20:
            terminate = True
            response = 'TOO LONG FAILED'
        else:
            terminate = False
        self.terminate = terminate
        return response, terminate


class DoctorSimulator(object):
    def __init__(self, decision_process, start_index, disease, llm_name):
        self.decision_process = decision_process
        self.start_index = start_index
        self.current_index = start_index
        self.disease = disease
        self.terminate = False
        self.conclusion = None
        self.llm_name = llm_name

        if self.disease == 'hf':
            self.disease_full_name = 'heart failure'
        else:
            assert ValueError('')

    def reset(self):
        self.current_index = self.start_index
        self.terminate = False
        self.conclusion = None

    def step(self, history):
        if len(history) > 0:
            response = history[-1]['response']
            _, next_condition = self.parse_response(response)
            terminate = self.proceed_decision_process(next_condition)
        else:
            terminate = False

        if not terminate:
            step_str = self.asking_question()
        else:
            step_str = self.conclusion
        self.terminate = terminate
        return step_str, terminate

    def asking_question(self):
        current_condition = self.decision_process[self.current_index]
        question = current_condition.condition
        prompt = ("Please assume you are a doctor specializing in cardiovascular diseases. "
                  "You are engaged in a {} diagnosis conversation and need to ask the following "
                  "question to a patient. Please only response question, Please strictly repose the same question"
                  "\n QUESTION: {}").format(self.disease_full_name, question)
        question_ask = call_llm(self.llm_name, prompt)
        return question_ask

    def parse_response(self, response):
        current_condition = self.decision_process[self.current_index]
        question = current_condition.condition
        prompt = ("Please assume that you are a doctor specializing in cardiovascular disease. "
                  "You are engaged in a {} diagnosis conversation. You have posed the following question "
                  "(described below) to a patient, and the patient has provided a response (also described below). "
                  "Please summarize the patient's response and reply with #YES# if they affirm the "
                  "question or #NO# if they negate the question.\n"
                  "QUESTION: {}\n\n RESPONSE: {}").format(self.disease_full_name, question, response)
        success_flag, parse_result = False, None
        while not success_flag:
            parse_result = call_llm(self.llm_name, prompt)
            if "#YES#" in parse_result or "#NO#" in parse_result:
                success_flag = True
        if "#YES#" in parse_result:
            return response, True
        else:
            return response, False

    def proceed_decision_process(self, next_condition):
        assert next_condition is True or next_condition is False
        current_condition = self.decision_process[self.current_index]
        if next_condition:
            next_step, step_type = current_condition.yes
        else:
            next_step, step_type = current_condition.no

        if step_type == 'DIAGNOSIS EXCLUDE' or step_type == 'DIAGNOSIS CONFIRM':
            assert isinstance(next_step, str)
            terminate = True
            self.conclusion = next_step
        else:
            assert step_type == 'NAVIGATE'
            assert isinstance(next_step, int)
            self.current_index = next_step
            terminate = False
        return terminate

