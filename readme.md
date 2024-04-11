# Conversational Diagnosis

This project contains all the relevant code to reproduce our work: [Conversational Disease Diagnosis via External Planner-Controlled Large Language Models](https://arxiv.org/abs/2404.04292).

We assume that users have the right to access MIMIC-IV Version 2.2. If not, please complete the necessary certification on PhysioNet to obtain access.

We also assume that users can access Microsoft Azure OpenAI as using the native OpenAI service may violate the data usage agreement of MIMIC-IV.

This repository was formed by merging two independent projects. Although we have conducted checks, there might be issues with the paths some scripts use to read files. If you encounter such issues, please contact us by sending an email to sunzj@zhejianglab.com or submitting a new issue in the repository.

## Step 1 Prepare API Key and Environment
Please add your own API key to empty_llm_util.py.

Then, please rename 'empty_llm_util.py' to 'llm_util.py'.

In this study, we default to using Azure OpenAI to call OpenAI models and the Baidu Qianfan platform to call Llama2.

Please run the project via python 3.12. Please install dependencies via requirements.txt

## Step 2 Preprocessing Dataset
You need to download the diagnoses_icd.csv and d_icd_diagnoses.csv files from MIMIC-IV, and discharge.csv and radiology.csv from MIMIC-IV-Note. Then, you need to place them in "./resource/mimic_iv/".
After that, run the scripts in the following order:

1. Run screening_agent/data_preprocess/get_admission_emr.py.
2. Run screening_agent/data_preprocess/text_embedding_generation.py.
3. Run convert_emr_to_symptom_list.py.

### Note:
1. The process of structuring electronic medical records in this work will incur some costs. It costs approximately 700 USD for 10,000 records (via GPT-4 Turbo). We are applying to upload the structured medical records to PhysioNet, so users will not need to structure them on their own in the future. 
2. We use text-embedding-large of open AI to obtain text embeddings, whose embedding size is 3072.


## Step 3 Training Screening Agent
Please run screening_agent/train.py.

You will obtain an inquiry model trained by reinforcement learning and a classifier trained by supervised learning.

Once the training process is completed, you can find the trained model saved in /resource/screening_model/ckpt.

## Step 4 Generating and Refining Decision Procedure
Run decision_procedure_generation.py to generate the initial decision procedure from the clinical guideline.

Then, run self_reflection.py to refine it manually.

The decision procedure refined by us is recorded in /resource/differential_diagnosis/heart_failure_decision_procedure_3.txt.

The automatically generated decision procedure should be similar to /resource/differential_diagnosis/heart_failure_decision_procedure_0.txt.

## Step 5 Screening Performance Evaluation
Run screening_simulation.py to conduct simulated conversations between patients and doctors.

You can set the max dialogue round, the backbone of doctors, the backbone of patients, and whether to utilize an external planner. However, it is recommended to use GPT-4 Turbo as the backbone for patients, as we found GPT-3.5 may miss important information.

You can evaluate screening performance by setting eval_model to true.

The screening result is recorded in our paper (Table 1).


## Step 6 Differential Diagnosis Performance Evaluation
Run decision_procedure_generation.py to evaluate the performance of the diagnosis, whose results are recorded in our paper (Figure 5).