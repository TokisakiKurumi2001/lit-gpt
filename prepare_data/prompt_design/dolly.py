import json
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from lit_gpt.tokenizer import Tokenizer
import re
import os
import pickle

ckpt = Path("/khoilm1/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf")
MAX_SEQ_LEN = 2048
maps = {
    'open_qa': 'Openbook Question answering',
    'creative_writing': 'Creative Writing',
    'closed_qa': 'Closed-book Question answering',
    'summarization': 'Summarization',
    'brainstorming': 'Brainstorming',
    'classification': 'Classification on text',
    'general_qa': 'General domain Question answering',
    'information_extraction': 'Information extracting',
}

def generate_inst(category: str) -> str:
    domain = maps[category]
    buffer = f"The following are questions (with answers) for {domain} task.\n"
    return buffer

def input_template(question: str, context: str) -> str:
    buffer = ""
    if len(context) != 0:
        buffer += f'Passage: \n{context}\n'
    buffer += f"Question: {question}\n"
    buffer += f"Answer: "
    return buffer
    

def output_template(answer: str) -> str:
    return f"{answer}"

def aggregate_prompt(main_input, shot_input, shot_output, category, few_shot_sep: str="\n\n"):
    res = generate_inst(category)
    res += few_shot_sep.join([f"{i}{o}" for i, o in zip(shot_input, shot_output)] + [main_input])
    return res

def check_valid(tokenizer, input_seq: str) -> bool:
    output = tokenizer.encode(input_seq, eos=True)
    return output.shape[0] < MAX_SEQ_LEN

if __name__ == "__main__":
    print("Loading tokenizer ...")
    tokenizer = Tokenizer(ckpt)

    print("Loading data ...")
    with open(f'dolly_fewshot.jsonl', 'r') as fin:
        json_data = re.sub(r"}\s*{", "},{", fin.read())
        sample_list = json.loads("["+json_data+"]")
        data_ls = sample_list

    res = []
    for data in tqdm(data_ls):
        main_sample = data['origin']
        main_input_seq = input_template(main_sample['question'], main_sample['context'])
        main_output_seq = output_template(main_sample['answer'])
        shot_input_seq = []
        shot_output_seq = []
        for shot in data['shot']:
            shot_input_seq.append(input_template(shot['question'], shot['context']))
            shot_output_seq.append(output_template(shot['answer']))

        num_shot = 5
        prompt = aggregate_prompt(main_input_seq, shot_input_seq[0:num_shot], shot_output_seq[0:num_shot], main_sample['category'])
        valid = check_valid(tokenizer, prompt + main_output_seq)
        while not valid:
            num_shot -= 1
            if num_shot == 0:
                valid = False
                break
            prompt = aggregate_prompt(main_input_seq, shot_input_seq[0:num_shot], shot_output_seq[0:num_shot], main_sample['category'])
            valid = check_valid(tokenizer, prompt + main_output_seq)
        
        if valid:
            res.append({'input': prompt, 'output': main_output_seq, 'instruction': ''})

    print(f"Accumulate {len(res)} samples.")

    if not os.path.exists('cache/dolly/'):
        os.makedirs('cache/dolly/')

    with open(f'cache/dolly/result_data.pkl', 'wb') as fout:
        pickle.dump(res, fout)

    with open('dolly_prompt_data.jsonl', 'w') as fout:
        for data in res:
            fout.write(json.dumps(data)+'\n')


    