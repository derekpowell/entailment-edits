import numpy as np
import pandas as pd
import torch
import re
import configparser

from openai import OpenAI

def openai_key():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["openai"]["key"]

## load data
ORACLE = True

if ORACLE:
    df = pd.read_json("https://raw.githubusercontent.com/eladsegal/strategyqa/refs/heads/main/data/strategyqa/dev.json")
    df = df[df['facts'].str.len() == 2]
    savetype = 'oracle'
else:
    df = pd.read_json("strategyqa-dev-modelcot.jsonl", orient='records', lines=True)
    savetype = 'model'

## define functions

def reverse_oracle_answer_prompt(row, oracle = True):
    question = row['question']
    if oracle:
        facts = row['facts']
        answer = row['answer']
    else:
        facts = [next_till_break(row['cot_answer'], r"(?<=1\.).*"), next_till_break(row['cot_answer'], r"(?<=2\.).*")]
        answer = row['cot_answer'].split('\n')[-1]
    
    # reasoning = "\n-".join(row['decomposition'])

    COT_start = f"1. {facts[0]}\n2. {facts[1]}\n"#The full set of logical reasoning steps is: \n-{reasoning}"
    pre_instr = "Consider the following question and chain of reasoning for its answer."
    question_prompt = f"Question: {question}\nAnswer: {answer}\nThe reasons for the answer are:\n{COT_start}"
    instr = "I am going to ask you to consider a question and chain of reasoning for its answer. Your task is to rewrite the question to have the opposite answer by changing only one of the subjects, so that the corresponding line of factual reasoning would thereby produce a different result. Do not change anything else about the question except the identity of one of it subjects. And remember, the new pattern of reasoning should run along the same lines as the original reasoning in the original question."
    oneshot = '''For example, if the question was: "Does Julius Caesar have more descendants than Ghengis Kahn?", one line of original reasoning might be:\n\n1. Ghengis Kahn fathered hundreds of children\n2. Julius Caesar fathered three children.\nAnswer: False.\n\nA rewritten question that makes the answer "True" might ask "Does Julius Caesar have more descendants than Isaac Newton?". In this case, one of the lines of reasoning could be that "Isaac Newton died a virgin".'''
    format_instr = "After your reasoning, finish by completing your answer in the following format:\n\n--==--\nOriginal answer: [original answer]\nNew answer: [opposite of original answer]\nOriginal Subject: [original subject]\nNew Subject: [new subject]\nNew Question: [new question]\nNew reasoning:\n1. [reason 1]\n2. [reason 2]"
    full_prompt = instr + " " + oneshot + '\n\n' + f"Now, here is the question to be rewritten:\n{question_prompt}"  + "\n" + "Rewrite the question above and explain the fact about the new subject that supports. Think step-by-step to accomplish this. First, identify the subjects to change by naming the original subjects and potential new subjects for each. Then, pick one to focus on for the rewrite, and rewrite the question changing only that subject. Finally, explain the new reasoning. " + format_instr
    
    return(full_prompt)

def next_till_break(text, query):
    match = re.search(query, text)
    if match:
        return(match.group().strip().strip('"'))  # Removes leading/trailing spaces

def get_matches(text):
    res = {
        "reverse_question": r"(?<=New Question:).*", 
        "reverse_answer": r"(?<=New answer:).*", 
        "reverse_term": r"(?<=New Subject:).*", 
        "original_term": r"(?<=Original Subject:).*", 
        "r1": r"(?<=1\.).*",
        "r2": r"(?<=2\.).*"
        }
    res = {k : next_till_break(text, v) for k, v in res.items()}
    res['reverse_facts'] = [res['r1'], res['r2']]
    
    return {k:v for k,v in res.items() if k not in ['r1', 'r2']}

def parse_completion(completion):
    text = completion.choices[0].message.content
    if "--==--" in text:
        formatted_response = text[text.index("--==--"):]
        matches = get_matches(formatted_response)
    else:
        matches = None
        
    return matches

## --- run

from tqdm import tqdm
client = OpenAI(api_key=openai_key())

new_data = []

for idx, item in tqdm(df.iterrows(), total=df.shape[0]):
    prompt = reverse_oracle_answer_prompt(item, ORACLE)
    completion = client.chat.completions.create(
        model= "gpt-4o-2024-08-06", #"o1-mini-2024-09-12",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=500
    )
    result = parse_completion(completion)
    if result:
        result['full_text'] = completion.choices[0].message.content
        result['qid'] = item['qid']
    else:
        result = {'full_text': completion.choices[0].message.content}
    
    new_data.append(result)
    
new_df = pd.DataFrame(new_data) # remove nones
new_df.to_json(f"gpt4-{savetype}-reversed.jsonl", orient = 'records', lines = True)