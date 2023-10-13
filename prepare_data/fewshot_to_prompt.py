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
TIGER-Lab/MathInstruct: question & answer & source
"""

def handle_each_math_format(dic):
    if 'Rationale' in dic.keys():
        replaced_dic = {
            'a )': '\tA.',
            'b )': '\tB.',
            'c )': '\tC.',
            'd )': '\tD.',
            'e )': '\tE.',
            ' , ': '\n'
        }
        option = dic['options']
        for k,v in replaced_dic.items():
            option = option.replace(k,v)
        ques = '\n'.join([dic['question'],option])
        ans = dic['Rationale'][:-1] + dic['Rationale'][-1:].upper()
    elif 'type' in dic.keys():
        ques = dic['question']
        ans = dic['answer']
    elif 'answer_option_list' in dic.keys():
        option = '\n\t'.join([': '.join(obj[0].values()) for obj in dic['answer_option_list']])
        ques = '\n\t'.join([dic['question'],option])
        ans = 'The answer is: '.join([dic['answer_analysis'][0], dic['answer']])
    elif 'source' in dic.keys():
        ques = dic['question']
        replaced_dic = {
            '(A)': '\n\tA.',
            '(B)': '\n\tB.',
            '(C)': '\n\tC.',
            '(D)': '\n\tD.',
            '(E)': '\n\tE.',
        }
        for k,v in replaced_dic.items():
            ques = ques.replace(k,v)
        ans = dic['answer']
    elif ['question', 'answer'] == list(dic.keys()):
        ques = dic['question']
        ans = dic['answer']
    else:
        raise ValueError('Dataset Not found. Value: ', dic)
    return ques, ans

# handle MMLU, OpenboolQA, TruthQA, BBQ
"""
MMLU: question & subject & choices & answer
OpenboolQA: question & id & choices & labels & answer
TruthQA: question & answer & type
BBQ: question & context & answer & choices & gold_index
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
    elif 'gold_index' in dic:
        ques = ' '.join((dic['context'], dic['question']))
        options = '\n\t'.join([f'{opt}. {ans}' for opt, ans in list(zip(['A','B','C'],dic['choices']))])
        ques = '\n\t'.join([ques, options])
        map_numb_char = {0:'A', 1:'B',2:'C'}
        ans = map_numb_char[dic['gold_index']]
    else:
        print(dic)
        raise ValueError()
    return ques, ans

# handle MedMCQA, Winogrande, BoolQ, SCIQ
"""
MedMCQA: id & question & opa & opb & opc & opd & answer & exp
Winogrande: question & option1 & option2 & answer
BoolQ: question & answer & passage
SCIQ: question, distractor1, distractor2, distractor3, answer, support
"""
def handle_each_complexqa1_format(dic):
    if 'exp' in dic:
        options = '\n\t'.join(
            [f'{opt}. {ans}' for opt, ans in list(
                zip(['A','B','C','D'],
                    (dic['opa'], dic['opb'], dic['opc'], dic['opd'])
                )
            )]
        )
        ques = ''.join((dic['question'],'?\n\t', options))
        ques = ques.replace('??',"?")
        map_numb_char = {0:'A', 1:'B',2:'C',3:'D'}
        if dic['exp']:
            ans = '.So the answer is '.join([dic['exp'],map_numb_char[dic['answer']]])
        else:
            ans = map_numb_char[dic['answer']]
    elif 'option1' in dic:
        options = '\n\t'.join(
            [f'{opt}. {ans}' for opt, ans in list(
                zip(['A','B'],
                    (dic['option1'], dic['option2'])
            ))]
        )
        ques = 'Fill the under-hypher. ' + dic['question'] + "\n\t" + options
        map_numb_char = {'1':'A', '2':'B'}
        ans = map_numb_char[dic['answer']]
    elif  'distractor1' in dic:
        options = '\n\t'.join(
            [f'{opt}. {ans}' for opt, ans in list(
                zip(['A','B','C','D'],
                    (dic['distractor1'], dic['distractor2'], dic['distractor3'], dic['answer'])
            ))]
        )
        ques = dic['question']
        ques = '\n\t'.join([ques,options])
        ans = dic['support']+ 'So the answer is D.'
    elif 'passage' in dic:
        ques = ' '.join([dic['passage'],dic['question']])
        ans = dic['answer']
    else:
        print(dic)
        raise ValueError()
    return ques, ans

def domain_func(domain, origin_shot_dic):
    len_threshold = 1500
    if domain == 'math':
        func = handle_each_math_format
    elif domain == 'science':
        func = handle_each_science_format
    elif domain == 'complexqa1':
        func = handle_each_complexqa1_format

    origin_ques, origin_ans = func(origin_shot_dic['origin'])
    origin_input = "Q: {}\nA:".format(origin_ques)
    if len(origin_input.split(' '))>len_threshold:
        return None #remove this question

    prompt = ''
    num_over_len = 0
    num_example = 0
    for exampler in origin_shot_dic['shot']:
        ques, ans = func(exampler)
        exampler_input = "Q: {}\nA: {}\n\n".format(ques, ans)
        if len((prompt + exampler_input + origin_input).split(' '))>len_threshold:
            num_over_len += 1
            break
        prompt += exampler_input
        num_example += 1
    if num_example == 1:
        print('only a example')
    prompt += origin_input

    res = {
        "instruction":f"The following are multiple choice questions (with answers).\n" if '\n\tA. ' in ques else "",
        "input":prompt,
        "output":origin_ans
    }
    return res, num_over_len


def main(domain_path, out_path, domain):
    res = []
    with open(domain_path, 'r') as fin:
        json_data = re.sub(r"}\s*{", "},{", fin.read())
        sample_list = json.loads("["+json_data+"]")
        num_over_lens = 0
        for sample in sample_list:
            inp, num_over_len = domain_func(domain, sample)
            if inp:
                num_over_lens += num_over_len
                res.append(inp)
    print('num_over_lens:',num_over_lens)
    print('Read {} samples, Write {} samples'.format(len(sample_list), len(res)))
    with open(out_path, 'w') as fout:
        fout.write(json.dumps(res, indent=4))

     
            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--domain', type=str, help='domain for handling', choices=['chat','math','science','cnn', 'complexqa1'], default='math')
    argparser.add_argument('--dir', type=str, help='directory containing <domain>_fewshot.jsonl', default='/vinai/khoilm1/vuongntm')
    argparser.add_argument('--out_path', type=str, help='path of output, the extension is json', default=None)

    args = argparser.parse_args()

    inp_path = os.path.join(args.dir, '{}_fewshot.jsonl'.format(args.domain))
    if not args.out_path:
        out_path = os.path.join(args.dir, '{}_fintune.jsonl'.format(args.domain))

    main(inp_path, out_path, args.domain)
    
