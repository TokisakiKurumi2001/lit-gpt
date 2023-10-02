import argparse
import os
import json
import re

# handle GSM8K, MathQA, MetaMath, TAL-SCQ5K
"""
GSM8K: question & answer
MathQA: question & answer & Rationale () & options
MetaMath: question & answer & type
TAL-SCQ5K: question & answer & answer_option_list & answer_analysis
"""

def handle_each_math_format(dic, is_example):
    if 'Rationale' in dic.keys():
        replaced_dic = {
            'a )': '\tA.',
            'b )': '\tB.',
            'c )': '\tC.',
            'd )': '\tD.',
            'e )'
            ' , ': '\n'
        }
        option = dic['options']
        for k,v in replaced_dic.items():
            option = option.replace(k,v)
        ques = '\n'.join([dic['question'],option])
        if not is_example:
            ans = dic['answer'].upper()
        else:
            ans = dic['Rationale'][:-1] + dic['Rationale'][-1:].upper()
    elif 'type' in dic.keys():
        ques = dic['question']
        if not is_example:
            trail = 'The answer is: '
            ans = dic['answer'][dic['answer'].find(trail) + len(trail):]
        else:
            ans = dic['answer']
    elif 'answer_option_list' in dic.keys():
        option = '\n\t'.join([': '.join(obj[0].values()) for obj in dic['answer_option_list']])
        ques = '\n\t'.join([dic['question'],option])
        if not is_example:
            ans = dic['answer']
        else:
            ans = 'The answer is: '.join([dic['answer_analysis'][0], dic['answer']])
    elif ['question', 'answer'] == list(dic.keys()):
        ques = dic['question']
        if not is_example:
            trail = '#### '
            ans = dic['answer'][dic['answer'].find(trail) + len(trail):]
        else:
            ans = dic['answer']
    else:
        raise ValueError('Dataset Not found. Value: ', dic)
    return ques, ans

# handle MMLU, OpenboolQA, TruthQA, BBQ
"""
MMLU: question & subject & choices & answer
OpenboolQA: question & id & choices & labels & answer
TruthQA: question & answer & type
BBQ: question & answer & answer_option_list & answer_analysis
"""
def handle_each_science_format(dic, is_example):
    if 'subject' in dic and 'choices' in dic:
        options = '\n\t'.join([f'{opt}. {ans}' for opt, ans in list(zip(['A','B','C','D'],dic['choices']))])
        ques = '\n\t'.join([dic['question'], options])
        map_numb_char = {0:'A', 1:'B',2:'C',3:'D'}
        ans = map_numb_char[dic['answer']]
    elif 'id' in dic and 'choices' in dic:
        options = '\n\t'.join([f'{opt}. {ans}' for opt, ans in list(zip(dic['choices']['label'],dic['choices']['text']))])
        ques = '\n\t'.join([dic['question'], options])
        ans = dic['answer']
    else:
        print(dic)
        raise ValueError()
    return ques, ans
        
len_threshold = 2000
def domain_func(func, origin_shot_dic):
    prompt = ''
    for exampler in origin_shot_dic['shot']:
        ques, ans = func(exampler, True)
        if len((prompt + "Q: {}\nA: {}\n\n".format(ques, ans)).split(' '))>2000:
            print('Cannot add examplers due to length limitation')
            break
        prompt += "Q: {}\nA: {}\n\n".format(ques, ans)
    ques, ans = func(origin_shot_dic['origin'], False)
    prompt += "Q: {}\nA:".format(ques)
    res = {
        "instruction":"The following are multiple choice questions (with answers) about math.\n" if '\n\tA. ' in ques else "",
        "input":prompt,
        "output":ans
    }
    return res


def main(domain_path, out_path, domain):
    res = []
    with open(domain_path, 'r') as fin:
        json_data = re.sub(r"}\s*{", "},{", fin.read())
        sample_list = json.loads("["+json_data+"]")
        for sample in sample_list:
            if domain=='math':
                res.append(domain_func(handle_each_math_format, sample))
            if domain=='science':
                res.append(domain_func(handle_each_science_format, sample))
    with open(out_path, 'w') as fout:
        fout.write(json.dumps(res, indent=4))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--domain', type=str, help='domain for handling', choices=['chat','math','science','cnn'], default='math')
    argparser.add_argument('--dir', type=str, help='directory containing <domain>_fewshot.jsonl', default='/vinai/khoilm1/vuongntm')
    argparser.add_argument('--out_path', type=str, help='path of output, the extension is json', default=None)
    
    args = argparser.parse_args()
    
    inp_path = os.path.join(args.dir, '{}_fewshot.jsonl'.format(args.domain))
    if not args.out_path:
        out_path = os.path.join(args.dir, '{}_fintune.jsonl'.format(args.domain))

    main(inp_path, out_path, args.domain)

