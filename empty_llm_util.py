import os
from logger import logger
import json
import requests
from openai import AzureOpenAI


os.environ['AZURE_OPENAI_KEY'] = ''
client = AzureOpenAI(
  azure_endpoint="https://zhejianglab.openai.azure.com/",
  api_key=os.getenv("AZURE_OPENAI_KEY"),
  api_version="2024-02-15-preview"
)


def call_open_ai_embedding(input_text):
    embedding = client.embeddings.create(
        model='', # replace to your own model name
        input=input_text).data[0].embedding
    return embedding


def call_open_ai(prompt, model_name):
    completion = client.chat.completions.create(
        model=model_name,  # model = "deployment_name"
        messages=prompt,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    response = completion.choices[0].message.content
    return response


def call_llm(llm_name, prompt):
    if 'gpt' in llm_name:
        response = call_open_ai(prompt, llm_name)
    elif 'llama2-70b' in llm_name:
        if len(prompt) > 8000:
            prompt = prompt[-8000:]
            logger.info('prompt truncated to 8000')
        response = call_llama2_70(prompt)
    elif llm_name == 'qwen_7':
        if len(prompt) > 8000:
            prompt = prompt[-8000:]
            logger.info('prompt truncated to 8000')
        response = call_qwen(prompt, 7)
    else:
        assert llm_name == 'qwen_14'
        if len(prompt) > 8000:
            prompt = prompt[-8000:]
            logger.info('prompt truncated to 8000')
        response = call_qwen(prompt, 14)
    return response


def call_llama2_70(prompt):
    access_token = ""
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_70b?access_token="
    url_access = url + access_token

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    success_flag = False
    response_utterance, content = '', None
    fail_time = 0
    while not success_flag:
        try:
            if fail_time > 10:
                break
            response = requests.request("POST", url_access, headers=headers, data=payload)
            assert response.status_code == 200
            content = response.content
            content = content.decode('utf-8')
            content = json.loads(content)
            response_utterance = content['result']
            success_flag = True
        except Exception as exp:
            fail_time += 1
            print('Error: {}, response: {}'.format(exp, content))

    return response_utterance


def call_qwen(prompt, model_size=14):
    url = 'http://10.5.29.170:8000/call_llm_Qwen_{}B'.format(model_size)
    headers = {'Content-Type': "application/json"}

    payload = json.dumps({'prompt': prompt})

    success_flag, retry_time = False, 0
    response = 'ERROR'
    while not success_flag:
        try:
            x = requests.post(url, headers=headers, data=payload)
            response = x.json()
            assert isinstance(response, str) and len(response) > 0
            success_flag = True
        except Exception as exp:
            print('Error: {}'.format(exp))
    return response
