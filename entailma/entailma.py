import pandas as pd
import transformers
import torch


from pathlib import Path



PREMISES_PROMPT_PATH = "prompt3b.txt"
QA_EXAMPLE_PATH = "data/obqa/dev.tsv"
QA_PROMPT_PATH = None

path = Path(__file__).parent / PREMISES_PROMPT_PATH

# with path.open() as f:
#     test = list(csv.reader(f))

with open(path, 'r') as file:
    premises_prompt = file.read()


def generate_qa_prompt(fname, n = 32):
    df = pd.read_csv(fname, sep='\t')
    plist = []
    for i in range(n):
        plist.append("Question: " + df.iloc[i]["Complete Question"] + "\nAnswer: " + df.iloc[i]["Answer Key"])

    return("\n".join(plist))

if QA_PROMPT_PATH:
    with open(QA_PROMPT_PATH, 'r') as file:
        mc_answer_prompt = file.read()
elif QA_EXAMPLE_PATH:
    mc_answer_prompt = generate_qa_prompt("data/obqa/dev.tsv")


def mc_choose_answer(question, model, tokenizer=None):
    if not tokenizer:
        tokenizer = model.tok
    
    input_str = mc_answer_prompt + f"\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_str, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    sequences = model.generate(input_ids = input_ids, max_new_tokens = 1)
    
    return tokenizer.decode(sequences[0])[-1]


def generate_premises(question, answer, model, tokenizer):
    
    input_str = f"\n\n{premises_prompt}Question: {question}\nAnswer: {answer}\n"

    pipe = transformers.pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        torch_dtype=torch.float16,
        # device_map="cuda",
        device = model.device,
        # use_auth_token = auth_token()
    )

    sequences = pipe(
        input_str,
        # do_sample=True,
        # top_k = 50, 
        num_beams = 5, # beam search may be better ...
        max_new_tokens=150,
        temperature = 0.7
    )
    
    generated_text = sequences[0]['generated_text']
    premises = generated_text[len(input_str):-1] 

    return premises.split("\n")[:2]

