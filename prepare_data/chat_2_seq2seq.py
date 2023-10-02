import json
from typing import List, Dict

def build_seq2seq(data: List[str], chat_data: List[Dict[str,str]], sep: str, turn_sep_token: str = "\n"):
    previous_chat = ""
    for i in range(len(data)):
        current_sequence = data[i]
        if i % 2 == 0:
            # even indexed prompt, always the prompter
            seq2seq_example = {'input': '', 'output': ''}
            seq2seq_example['input'] = f"{previous_chat}{turn_sep_token}User: {current_sequence}{sep}Response: "
        else:
            # odd indexed, belongs to assistant
            current_sequence = data[i]
            seq2seq_example['output'] = current_sequence
            chat_data.append(seq2seq_example)
            previous_chat = f"{previous_chat}{seq2seq_example['input']}{current_sequence}"

if __name__ == "__main__":
    num_samples = 30000 # 10K
    chat_data = [] # [{'input': '', 'output': ''} ...]
    sep = '\n'

    with open('chat_data.jsonl') as fin:
        for line in fin:
            data = json.loads(line)
            data = data['chat']
            build_seq2seq(data, chat_data, sep)

    with open('chat_sample_10K.jsonl', 'w') as fout:
        for i in range(num_samples):
            _data = chat_data[i]
            # fout.write(json.dumps(_data, indent=2, default=str)+'\n')
            fout.write(json.dumps(_data)+'\n')

            

            