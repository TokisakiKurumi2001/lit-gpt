import json
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from lit_gpt.tokenizer import Tokenizer
import re
import os
import pickle

MAX_SEQ_LEN = 2048

def input_template(article: str, input_prefix: str = "###\nArticle: ", input_suffix: str = "\n\n", output_command: str = "Summarize the above article in 3 sentences.") -> str:
    return f"{input_prefix}{article}{input_suffix}{output_command}"

def output_template(summary: str, output_prefix: str = "\n") -> str:
    return f"{output_prefix}{summary}"

def aggregate_prompt(main_input, shot_input, shot_output, few_shot_sep: str="\n\n"):
    res = few_shot_sep.join([f"{i}{o}" for i, o in zip(shot_input, shot_output)] + [main_input])
    return res

def check_valid(tokenizer, input_seq: str) -> bool:
    output = tokenizer.encode(input_seq, eos=True)
    return output.shape[0] < MAX_SEQ_LEN

if __name__ == "__main__":
    print("Loading tokenizer")
    ckpt = Path("/khoilm1/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf")
    tokenizer = Tokenizer(ckpt)

    res = []
    print("Loading data ...")
    with open('cnn_fewshot.jsonl') as fin:
        json_data = re.sub(r"}\s*{", "},{", fin.read())
        sample_list = json.loads("["+json_data+"]")
    
    for data in tqdm(sample_list):
        main_sample = data['origin']
        main_input_seq = input_template(main_sample['question'])
        main_output_seq = output_template(main_sample['answer'])
        shot_input_seq = []
        shot_output_seq = []
        for shot in data['shot']:
            shot_input_seq.append(input_template(shot['question']))
            shot_output_seq.append(output_template(shot['answer']))

        num_shot = 5
        prompt = aggregate_prompt(main_input_seq, shot_input_seq[0:num_shot], shot_output_seq[0:num_shot])
        valid = check_valid(tokenizer, prompt + main_output_seq)
        while not valid:
            num_shot -= 1
            if num_shot == 0:
                valid = False
                break
            prompt = aggregate_prompt(main_input_seq, shot_input_seq[0:num_shot], shot_output_seq[0:num_shot])
            valid = check_valid(tokenizer, prompt + main_output_seq)
        
        if valid:
            res.append({'input': prompt, 'output': main_output_seq, 'instruction': ''})

    print(f"Accumulate {len(res)} samples.")

    if not os.path.exists('cache/cnn/'):
        os.makedirs('cache/cnn/')

    with open(f'cache/cnn/result_data.pkl', 'wb') as fout:
        pickle.dump(res, fout)

    with open('cnn_prompt_data.jsonl', 'w') as fout:
        for data in res:
            fout.write(json.dumps(data)+'\n')
