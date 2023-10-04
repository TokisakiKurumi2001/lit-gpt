import json
from tqdm import tqdm
from pathlib import Path
from lit_gpt.tokenizer import Tokenizer
import pickle
import os

MAX_SEQ_LEN = 2048

def check_valid(tokenizer, input_seq: str) -> bool:
    output = tokenizer.encode(input_seq, eos=True)
    return output.shape[0] < MAX_SEQ_LEN

if __name__ == "__main__":
    print("Loading tokenizer")
    model = 'mistral' # llama2, mistral
    if model == 'llama2':
        ckpt = Path("/khoilm1/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf")
    else:
        ckpt = Path("/khoilm1/lit-gpt/checkpoints/mistralai/Mistral-7B-v0.1")
    tokenizer = Tokenizer(ckpt)

    res = []
    with open(f'{model}_chat_sample.jsonl') as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            # data['instruction'] = 'The following are conversational chats with the user. Be a helpful assistant and response.'
            data['instruction'] = ''
            valid = check_valid(tokenizer, data['instruction'] + data['input'] + data['output'])
            if valid:
                res.append(data)

    print(f"Accumulate {len(res)} samples.")

    if not os.path.exists(f'cache/chat/{model}'):
        os.makedirs(f'cache/chat/{model}')

    with open(f'cache/chat/{model}/result_data.pkl', 'wb') as fout:
        pickle.dump(res, fout)

    with open(f'{model}_chat_prompt_data.jsonl', 'w') as fout:
        for data in res:
            fout.write(json.dumps(data)+'\n')