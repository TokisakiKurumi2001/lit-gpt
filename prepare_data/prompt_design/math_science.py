import json
from tqdm import tqdm
from pathlib import Path
from lit_gpt.tokenizer import Tokenizer
import pickle
import os
import re

MAX_SEQ_LEN = 2048

def check_valid(tokenizer, input_seq: str) -> bool:
    output = tokenizer.encode(input_seq, eos=True)
    return output.shape[0] <= MAX_SEQ_LEN

if __name__ == "__main__":
    print("Loading tokenizer")
    ckpt = Path("/khoilm1/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf")
    tokenizer = Tokenizer(ckpt)

    domain = 'science'
    res = []
    with open(f'{domain}_fintune.jsonl', 'r') as fin:
        json_data = re.sub(r"}\s*{", "},{", fin.read())
        sample_list = json.loads("["+json_data+"]")
        data_ls = sample_list[0]

    for data in tqdm(data_ls):
        valid = check_valid(tokenizer, data['instruction'] + data['input'] + data['output'])
        if valid:
            res.append(data)

    print(f"Accumulate {len(res)} samples.")

    if not os.path.exists(f'cache/{domain}/'):
        os.makedirs(f'cache/{domain}/')

    with open(f'cache/{domain}/result_data.pkl', 'wb') as fout:
        pickle.dump(res, fout)

    with open(f'{domain}_prompt_data.jsonl', 'w') as fout:
        for data in res:
            fout.write(json.dumps(data)+'\n')