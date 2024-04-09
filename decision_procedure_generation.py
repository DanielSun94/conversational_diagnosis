from llm_util import call_open_ai
from config import knowledge_origin_text_path, diagnosis_procedure_init_path


def main():
    disease = 'heart failure'
    with open(knowledge_origin_text_path, 'r', encoding='utf-8-sig', newline='') as f:
        content = ''.join(f.readlines())
    result = generate_decision_procedure(disease, content)
    with open(diagnosis_procedure_init_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(result)


def generate_decision_procedure(disease, content):
    prompt = (("(1) Please generate a decision procedure to describe the diagnosis of {} step by step "
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
               "Each condition must have \"YES\" or \"NO\" as the two branches; no other branches are allowed.\n"
               "The CLINICAL GUIDELINE:\n{}")
              .format(disease, disease, disease, content))
    result = call_open_ai(prompt, 'gpt_4_turbo')
    return result


if __name__ == '__main__':
    main()
