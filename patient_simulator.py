from llm_util import call_llm


class PatientSimulator(object):
    def __init__(self, patient_llm_name):
        assert patient_llm_name == 'gpt_4_turbo' or patient_llm_name == 'gpt_4o'
        self.current_key = None
        self.current_data = None
        self.llm_name = patient_llm_name
        self.index_pointer = 0

    def response(self, system_utterance):
        prompt = ("Please presume you are a patient; The doctor is asking you a question. The question is: {}\n\n"
                  "Please answer the doctor's question according to the content below, "
                  "which is the clinical note from your hospitalization.\n"
                  "NOTE: Please answer briefly and concisely. If the question can be answered with yes or no, "
                  "please respond with yes or no first. Do not provide information that the doctor did not ask for. "
                  "If you cannot find relevant content to answer the question, please respond that "
                  "the asked-about physical condition is normal.\n"
                  "Pay attention to abbreviations; for example, LVEF (left ventricular ejection fraction) "
                  " be written as EF.\n "
                  "CLINICAL NOTE:\n{}").format(system_utterance, self.current_data).strip()
        response = call_llm(self.llm_name, prompt)
        return response

    def reset(self, key, data):
        self.current_key = key
        self.current_data = data
        return self.current_key, self.current_data

