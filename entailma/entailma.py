import pandas as pd
import transformers
import torch
import torch.nn.functional as F
from easyeditor import LoRAHyperParams, FTHyperParams, BaseEditor

from pathlib import Path
import re


PREMISES_PROMPT_PATH = "entailer-dev-prompt-tandf.txt"
QA_EXAMPLE_PATH = "data/obqa/dev.tsv"
QA_PROMPT_PATH = None
path = Path(__file__).parent


# with path.open() as f:
#     test = list(csv.reader(f))

with open(path / PREMISES_PROMPT_PATH, 'r') as file:
    premises_prompt = file.read()


def generate_qa_prompt(fname, n = 32):
    df = pd.read_csv(fname, sep='\t')
    plist = []
    for i in range(n):
        plist.append("Question: " + df.iloc[i]["Complete Question"] + "\nAnswer: " + df.iloc[i]["Answer Key"])

    return("\n".join(plist))


def answer_choice_list(choices):
    options = re.split(r'\s*\(\w\)\s*', choices)
    return( [option.strip() for option in options if option] )


def generate_qa_cloze_prompt(fname, n = 32):
    df = pd.read_csv(fname, sep='\t')
    plist = []
    for i in range(n):
        ans = df.iloc[i]["Answer Key"]
        ans_ind = ['A','B','C','D'].index(ans)
        ans_text = answer_choice_list(df.iloc[i].Choices)[ans_ind]
        plist.append("Question: " + df.iloc[i]["Complete Question"] + "\nAnswer: " + ans_text)

    return("\n".join(plist))


if QA_PROMPT_PATH:
    with open(path / PREMISES_PROMPT_PATH, 'r') as file:
        mc_answer_prompt = file.read()

elif QA_EXAMPLE_PATH:
    mc_answer_prompt = generate_qa_prompt("data/obqa/dev.tsv")
    mc_answer_cloze_prompt = generate_qa_cloze_prompt("data/obqa/dev.tsv")


# def mc_choose_answer(question, model, tokenizer=None):
#     if not tokenizer:
#         tokenizer = model.tok
    
#     input_str = mc_answer_prompt + f"\nQuestion: {question}\nAnswer:"
#     inputs = tokenizer(input_str, return_tensors="pt")
#     input_ids = inputs["input_ids"].cuda()
#     sequences = model.generate(input_ids = input_ids, max_new_tokens = 1)
    
#     return tokenizer.decode(sequences[0])[-1]


def last_token_logprobs(text, last_token, model):
    x = model.logprobs(text)
    logprobs = x['logprobs']
    t_idx = [i[-1] for i in model.tok(last_token)['input_ids']]

    return(logprobs[0, -1, t_idx])


def mc_answer_logprobs(question, model, answers = ['A','B','C','D']):

    input_str = mc_answer_prompt + f"\n\nQuestion: {question}\nAnswer: "
    lt = last_token_logprobs(input_str, answers, model)

    return F.log_softmax(lt, -1)


def mc_choose_answer(question, model, answers = ['A','B','C','D']):
    lp = mc_answer_logprobs(question, model, answers)
    return answers[torch.argmax(lp)]


def mc_cloze_logprobs(question, ans_text, model):
    choices = answer_choice_list(ans_text)
    tok = model.tok
    model = model
    prompt = f"{mc_answer_cloze_prompt}\nQuestion: {question} Answer:"
    
    if type(tok) == transformers.models.llama.tokenization_llama.LlamaTokenizer:
        padded_choices = choices
        prompt = prompt + " " if prompt[-1]!= " " else prompt
    else:
        padded_choices = [pad_token(c) for c in choices] # pad all the 
    
    prompts = [prompt + c for c in choices]
    # print(prompts)
    logits = torch.tensor([model.completion_logprob(x[0], x[1]) for x in zip(prompts, choices)])
    
    return(F.log_softmax(logits, -1))


def completion_prob(preprompt, question, target_answer, model, answers = ['A','B','C','D']):
   if len(preprompt) == 0:
      prompt = mc_answer_prompt +  "\n\n" + preprompt + "Question:" + question + "\nAnswer: "
   else:
      prompt = mc_answer_prompt + '\n\n' + preprompt + '\nQuestion: ' + question  + '\nAnswer: '
   
   logprobs0 = last_token_logprobs(prompt, answers, model)
   prob = logprobs0[answers.index(target_answer)].exp() / logprobs0.exp().sum()

   return prob


def score_premises(premises, question, target_answer, model, base_prob = None, answers = ['A','B','C','D']):
   '''Returns the odds-ratio of the target answer with vs without the premises in the premises in the context.'''
   
   if not base_prob:
      base_prob = completion_prob("", question, target_answer, model, answers)

   premise_str = "\n".join(premises)
   prob1 = completion_prob(premise_str, question, target_answer, model, answers)

   return( (prob1/(1-prob1)) / (base_prob/(1-base_prob)))


def check_repeat_words(text1, text2, max_repeat_size = 4):
    # check if text2 includes a repetition of more than max_repeat_size in a row

    t1 = text1.lower().split()
    t2 = text2.lower().split()

    sublists = []
    for idx in range(len(t1) - max_repeat_size+1):
        s = t1[idx:idx+max_repeat_size+1]
        if len(s) > max_repeat_size:
            sublists.append(' '.join(s))
    
    sublists2 = []
    for idx in range(len(t2) - max_repeat_size+1):
        s = t2[idx:idx+max_repeat_size+1]
        if len(s) > max_repeat_size:
            sublists2.append(' '.join(s))
    
    valid = True

    for seq in sublists2:
        if seq in sublists:
            valid = False
            break

    return(valid)


def generate_premises(question, answer, model, num_prem = 1, batch_size = 4):
    
    input_str = f"{premises_prompt}\n\nQuestion: {question}\nAnswer: {answer}\n"

    pipe = transformers.pipeline(
        "text-generation",
        model = model.model,
        tokenizer = model.tok,
        torch_dtype=torch.float16,
        # device = model.model.device
    )

    seq_list = []
    for i in range(-(-num_prem // batch_size)):


        sequences = pipe(
            input_str,
            do_sample = True,
            top_p = .6,
            # penalty_alpha = 0.6, # avoids repetition of the question + answer (except doesn't)
            temperature = 0.7,
            max_new_tokens = 50,
            num_return_sequences = min(batch_size, num_prem - i*batch_size)
        )

        seq_list += sequences
    
    generated_texts = [s['generated_text'] for s in seq_list]
    
    premises = [t[len(input_str):-1] for t in generated_texts]
    premlist = [p.split("\n")[:2] for p in premises] 

    return premlist if len(premlist) > 1 else premlist[0]


def generate_best_premises(question, answer, model, num_prem=10, batch_size = 4):
    premises = generate_premises(question, answer, model, num_prem)
    premise_validity = [check_repeat_words(question, '\n'.join(p)) for p in premises]

    valid_premises = [i for (i, v) in zip(premises, premise_validity) if v]

    if len(valid_premises) > 0:

        base_completion_prob = completion_prob("", question, answer, model)
        scores =  [score_premises(p, question, answer, model, base_prob = base_completion_prob) for p in valid_premises]
        max_idx = scores.index(max(scores))
        
        return valid_premises[max_idx], scores[max_idx]
    
    else:
        return ["",""], -100.


class WrappedModel:
    def __init__(self, model, tokenizer, auth_token=None):
        
        self.model = model
        self.tok = tokenizer
        self.tok.pad_token_id = self.tok.eos_token_id
        # self.model_name = self.editor.model_name

        # self.params = hparams
        self.preprompt = ""
        self.saved_weights = None
        
        if type(self.tok) == transformers.LlamaTokenizer or transformers.LlamaTokenizerFast:
            self.tok.padding_side = "right"
        else: 
            self.tok.padding_side = "left"
    
    def edit(self, rewrite, log_file = None, **kwargs):
        if log_file:
            h = open(log_file, "a")
        else:
            h = None
        
        if "preprompt" in rewrite: # this is a little hacky
            self.preprompt = rewrite["preprompt"]
            return None
        
        else:
            with redirect_stdout(h): # None
                metrics, self.model, self.saved_weights = self.editor.pure_edit( # pure_edit
                    **rewrite,
                    # **kwargs,
                    keep_original_weight = True,
                    verbose = False
                )
        
        return metrics
    
    
    def restore(self):

        self.preprompt = ""
        
        if self.params.alg_name == "LoRA":
            self.model = self.model.unload()
        
        elif self.saved_weights:

            try:
                with torch.no_grad():
                    for k, v in self.saved_weights.items():
                        nethook.get_parameter(self.model, k)[...] = v
                self.saved_weights = None
                # print("Original model restored")
            except NameError as e:
                print(f"No model weights to restore: {e}")

        elif self.saved_weights == {}:
            print (print(f"No model weights to restore: saved_weights is empty dict"))

        return None

            
    def generate_text(self, texts, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if type(texts) != list:
            texts = [texts]
        
        texts = [self.preprompt + t for t in texts]

        model = self.model
        tokenizer = self.tok
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding, **kwargs) # 

            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
        return(generated_texts)
    
    
    # def logprobs(self, texts):
        
    #     # texts = self.preprompt + texts if type(texts)==str else [self.preprompt + t for t in texts]
    
    #     # tokenizer = self.tok 
    #     # model = self.model
    #     # encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

    #     # with torch.no_grad():
    #     #     model_out = model(encoding["input_ids"])
    #     #     logits = model_out.logits
    #     #     logprobs = F.log_softmax(logits, -1)

    #     x = self.logits(texts)
        
    #     return {"tokens": x['tokens'], "logprobs": logprobs}
    

    def logits(self, texts):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        texts = self.preprompt + texts if type(texts)==str else [self.preprompt + t for t in texts]
    
        tokenizer = self.tok 
        model = self.model
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_out = model(encoding["input_ids"])
            logits = model_out.logits
        
        return {"tokens": encoding, "logits": logits}
    
    
    def logprobs(self, texts):
        
        logits = self.logits(texts)
        
        return {"tokens": logits['tokens'], "logprobs": F.log_softmax(logits['logits'], -1)}
    
    
    def obs_logits(self, text):
    
        x = self.logits(text)
        logits = x['logits']
        
        obslogits = []

        if type(text) is str:
            tok_idx = x['tokens']['input_ids'].squeeze()
            logits = x['logits']
            obslogits = logits[0, :, tok_idx[1:]].squeeze().diag()

        elif type(text) is list:
            for i in range(len(text)):
                tok_idx = x['tokens']['input_ids'][i].squeeze()
                mask = x['tokens']['attention_mask'][i] > 0
                
                obslogits.append(logits[0, :, tok_idx[1:]].squeeze().diag()[mask[1:]])

        return obslogits


    def obs_logprobs(self, text):
        logits = self.obs_logits(text)

        return [F.log_softmax(l, -1) for l in logits] if type(logits)==list else F.log_softmax(logits, -1)
        
       
    def completion_logprob(self, text, completion, start_ind = None):
        
        '''
        Compute model log probability of completion substring. Returns single value tensor. Takes only one text string.
        '''

        return self.substring_logprobs(text, completion)[0][-1]
        

    def substring_logprobs(self, texts, substring, pad = True):
        '''
        Compute model log probability of each occurrence of substring in text. Returns list of list-type. Accepts a list of strings.
        '''
        
        if type(texts) != list:
            texts = [texts]
        
        logprobs = self.logprobs(texts)
        
        tok_encoded = encode_token(substring, self.tok, pad = pad)
        # text_encoded = logprobs['tokens']['input_ids'][0].tolist()
        
        out = []
        for i in range(len(texts)):
            text_encoded = logprobs['tokens']['input_ids'][i].tolist()

            # find matches for searched token sequence
            start_idxs = []
            for left in range(0, len(text_encoded) - len(tok_encoded)+1):
                # left = i - 1
                right = left + len(tok_encoded)
                if text_encoded[left:right] == tok_encoded:
                    start_idxs.append(left)

            lp = logprobs['logprobs'][i]
            match_probs = []

            # compute probability for all tokens
            for start in start_idxs:
                val = 0
                for i in range(len(tok_encoded)):
                    val += lp[start + i - 1][tok_encoded[i]]
                match_probs.append(val)

            out.append(match_probs)

        return out
        

    def choose(self, prompt, choices, normalization = None):

        # prompt = prompt.rstrip() # remove any trailing whitespace

        if type(self.tok) == transformers.models.llama.tokenization_llama.LlamaTokenizer:
            padded_choices = choices
            prompt = prompt + " " if prompt[-1]!= " " else prompt
        else:
            padded_choices = [pad_token(c) for c in choices] # pad all the 
        
        prompts = [prompt + c for c in padded_choices]

        logits = torch.tensor([self.completion_logprob(prompts[i], padded_choices[i]) for i in range(len(padded_choices))])

        if normalization == "unconditional":
            norm_logits = torch.tensor([self.completion_logprob(padded_choices[i], padded_choices[i]) for i in range(len(padded_choices))])
            logits = logits - norm_logits

        elif normalization == "byte_length":    
            str_lens = [len(c) for c in choices]
            logits = logits / torch.tensor(str_lens)

        elif normalization == "token_length":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = logits / torch.tensor(tok_lens)

        elif normalization == "root":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = torch.pow(torch.exp(logits), 1./torch.tensor(tok_lens))

        logits = logits.tolist()

        return(logits.index(max(logits)))