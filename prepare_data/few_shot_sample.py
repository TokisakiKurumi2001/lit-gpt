"""
Format will be JSONL with
{'origin': original sample, 'shots': [sample_1, sample_2, ..., sample_n]}
"""

from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import json
import torch
import pickle
from tqdm import tqdm

def export_data(data_path, *args, **kwargs):
    if data_path == 'meta-math/MetaMathQA':
        dataset = load_dataset(data_path, *args, **kwargs)
        dataset = dataset['train'].train_test_split(test_size=0.25, seed=42)

        return dataset['train']
    else:
        dataset = load_dataset(data_path, *args, **kwargs)
    
        return dataset

def save_few_shot_data(res, path):
    with open(path, 'w') as fout:
        for ques in res:
            fout.write(json.dumps(ques)+'\n')

def create_few_shot(model, domain_data_paths, query_id, result_path, cache_dir: str="", verbose=True):
    #create corpus
    corpus_sentences_dict = {}
    for local_path, ques_col_name, ans_col_name, args, kwargs in domain_data_paths:
        # dataset = load_dataset(local_path, *args, **kwargs)
        dataset = export_data(local_path, *args, **kwargs) # some datasets do not have test split --> split to select only train set
        if ques_col_name != 'question':
            dataset = dataset.rename_column(ques_col_name, 'question')
        if ans_col_name != 'answer':
            dataset = dataset.rename_column(ans_col_name, 'answer')
        print(dataset)
        corpus_sentences_dict.update(dict(zip(dataset['question'], dataset)))
    corpus_sentences = list(corpus_sentences_dict.keys())
    if verbose:
        print('Done to create Corpus which contains {} questions'.format(len(corpus_sentences)))

    # encode our corpus
    if verbose:
        print("Encode the corpus. This might take a while")
    if len(cache_dir) == 0:
        corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=verbose, convert_to_tensor=True)
        torch.save(corpus_embeddings, 'cache/corpus_embeddings.pt')
    else:
        print("Found cache, loading embeddings ...")
        corpus_embeddings = torch.load(f'{cache_dir}/corpus_embeddings.pt')

    res = []
    if verbose:
        print('Find shots ...')
    for sample_id in tqdm(query_id):
        # cluster = [id x N]

        queries = [corpus_sentences[sentence_id] for sentence_id in sample_id] # queries=['question' x N]
        query_embedding = model.encode(queries, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=6)
        
        for query, hits_query in zip(queries, hits):
            example = {}
            example['origin'] = corpus_sentences_dict[query]
            example['shot'] = [corpus_sentences_dict[corpus_sentences[hit['corpus_id']]] for hit in hits_query[1:]]
            res.append(example)

    save_few_shot_data(res, result_path)

if __name__ == "__main__":
    # Load model
    print('Loading model')
    ckpt = 'sentence-transformers/all-mpnet-base-v2'
    model = SentenceTransformer(ckpt)

    # domain_data_paths = [
    #     ['gsm8k','question', 'answer', ["main"], {"split":"train"}], 
    #     ['math_qa','Problem', 'correct',[], {"split":"train"}], 
    #     ['math-eval/TAL-SCQ5K','problem', 'answer_value', [], {'data_dir':"TAL-SCQ5K-EN","split":"train"}],
    #     ['meta-math/MetaMathQA', 'query', 'response', [], {}]
    # ]

    domain_data_paths = [
        ['cnn_dailymail', 'article', 'highlights', ['3.0.0'], {"split": 'train'}]
    ]

    # domain_data_paths = [
    #     ['lighteval/mmlu', 'question', 'answer', ['all'], {"split": 'auxiliary_train'}],
    #     ['lighteval/bbq_helm', 'question', 'references', ['all'], {"split": 'train'}],
    #     ['openbookqa', 'question_stem', 'answerKey', ['main'], {'split': 'train'}],
    # ]

    print('Loading sample id')
    cache_dir = 'cache/cnn'
    with open(f'{cache_dir}/sample_id.pkl', 'rb') as fin:
        sample_id = pickle.load(fin)

    print('Creating data')
    create_few_shot(model, domain_data_paths, sample_id, 'cnn_fewshot.jsonl', cache_dir=cache_dir)

