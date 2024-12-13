import configparser
def auth_token():

    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["hugging_face"]["token"]

def scratch_path():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return "/scratch/" + config["user"]["username"] + "/"

import os
if os.path.isdir(scratch_path()):
    os.environ['HF_HOME'] = scratch_path() + '.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = scratch_path() + '.cache/huggingface/datasets'
print(os.getenv('HF_HOME'))
print(os.getenv('HF_DATASETS_CACHE'))

## ---------------------------------------------------------------------
## Load libraries
## ---------------------------------------------------------------------

import numpy as np
import pandas as pd

import torch
import transformers
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch.nn.functional as F

import datasets

from tqdm import tqdm

## ---------------------------------------------------------------------
## Ensure GPU is available -- device should == 'cuda'
## ---------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

## ---- load model

# Load tokenizer and model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, device = device)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map= device)

## ----- load data

import pandas as pd

## answering original Qs with COT reasoning
df = pd.read_json("https://raw.githubusercontent.com/eladsegal/strategyqa/refs/heads/main/data/strategyqa/dev.json")
df = df[df['facts'].str.len() == 2]


## to answer reversed questions with COT reasoning
df = pd.read_json("gpt4-oracle-reversed.jsonl", orient='records', lines=True)
df['question'] = df['reverse_question']
df['answer'] = df['reverse_answer']
df['facts'] = df['reverse_facts']

## ----- get_answers
def cot_answer(row):
    question = row['question']
    
    messages = [
        {"role": "system", "content": "You are knowledgable assistant."},
        {"role": "user", "content": f"Answer the following questions true/false questions by reasoning step-by-step. Each question requires two steps of reasoning. You will make your response by giving two numbered reasons followed by a single word 'True' or 'False'. Each of your reasons should clearly and explicitly refer to one and only one of the main subjects mentioned in the question itself. Your entire response must follow this exact format:\n1. [first reason]\n2. [second reason]\n[True/False]\nNo other text, greetings, explanations, or question repetition is allowed. Think step by step and answer the following question:\nQuestion:{question}"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    #print(tokenizer.decode(response, skip_special_tokens=True))
    return tokenizer.decode(response, skip_special_tokens=True)


cot_answers = []
for idx, item in tqdm(df.iterrows(), total=df.shape[0]):
    cot_answers.append(cot_answer(item))

df['cot_answer'] = cot_answers

df.to_json("strategyqa-dev-reverse-modelcot.jsonl", orient = 'records', lines = True)