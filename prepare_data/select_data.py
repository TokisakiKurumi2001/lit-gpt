# based on https://www.sbert.net/examples/applications/clustering/README.html, using Fast Clustering
# code based on https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py

from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import numpy as np
import time
import json
import torch
import pickle
import os
import random
import itertools
import argparse

def export_data(data_path, *args, **kwargs):
    if data_path == 'meta-math/MetaMathQA':
        dataset = load_dataset(data_path, *args, **kwargs)
        dataset = dataset['train'].train_test_split(test_size=0.25, seed=42)

        return dataset['train']
    else:
        dataset = load_dataset(data_path, *args, **kwargs)
    
        return dataset

def create_clusters(PRETRAINED_EMBED_MODEL, domain_data_paths, min_community_size, threshold, override_path, cache_dir, force_rebuild, verbose=True):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer(PRETRAINED_EMBED_MODEL)

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
    if os.path.exists(f'{cache_dir}/corpus_embeddings.pt') and not force_rebuild['corpus_embeddings']:
        if verbose:
            print("Found cache, loading corpus embeddings")
        corpus_embeddings = torch.load(f'{cache_dir}/corpus_embeddings.pt')
    else:
        corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=verbose, convert_to_tensor=True)
        torch.save(corpus_embeddings, f'{cache_dir}/corpus_embeddings.pt')


    if verbose:
        print("Start clustering")
        start_time = time.time()

    #Two parameters to tune:
    #min_cluster_size: Only consider cluster that have at least 25 elements
    #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    if os.path.exists(f'{cache_dir}/cluster.pkl') and not force_rebuild['cluster']:
        if verbose:
            print("Found cache, loading cluster")
        with open(f'{cache_dir}/cluster.pkl', 'rb') as file:
            clusters = pickle.load(file)
    else:
        clusters = util.community_detection(
            corpus_embeddings, 
            min_community_size=min_community_size, 
            threshold=threshold
        )
        with open(f'{cache_dir}/cluster.pkl', 'wb') as file:
            pickle.dump(clusters, file)

    # calculate sum of number of element in each cluster
    total_num = 0
    for cluster in clusters:
        total_num += len(cluster)
    print(f'There are in total {total_num} elements in cluster')
    
    cluster_dic = {}
    for i, cluster in enumerate(clusters):
        cluster_dic[i] = [corpus_sentences_dict[corpus_sentences[sentence_id]] for sentence_id in cluster]
    
    if verbose:
        print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    # save cluster to json
    with open(override_path, 'w') as fout:
        fout.write(json.dumps(cluster_dic, indent=2, default=str))
    if verbose:
        print('Saved New Clusters.')

    # Check Cluster by Printing for all clusters the top 3 and bottom 3 elements
    # if verbose:
    #     for i, cluster in enumerate(clusters):
    #         print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    #         for sentence_id in cluster[0:3]:
    #             print("\t", corpus_sentences[sentence_id])
    #         print("\t", "...")
    #         for sentence_id in cluster[-3:]:
    #             print("\t", corpus_sentences[sentence_id])
    return {"sample_dict": cluster_dic, "id_dict": clusters}

def load_clusters(path):
    with open(path) as fin:
        cluster_dic = json.load(fin)
    return cluster_dic

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def select_samples(num_samples, clusters_dictionary, cache_dir):
    # !!!important!!! make sure that you called set_seed
    clusters = clusters_dictionary["id_dict"]
    clusters_dict = clusters_dictionary["sample_dict"]
    
    clusers_len = np.array([len(c) for c in clusters]) # array of length (= number of element) of each cluster
    cluster_num_sample = clusers_len * num_samples / sum(clusers_len)
    cluster_num_sample = np.where(cluster_num_sample<1,1,cluster_num_sample.astype(int)) # array stores number of element in each cluster should be sampled

    res = []
    ids = []
    for i, cluster in enumerate(clusters):
        examples = clusters_dict[i]

        sample_index = random.sample(list(range(len(cluster))), min(cluster_num_sample[i], len(cluster)))
        
        ids.append([cluster[i] for i in sample_index])
        res.append([examples[i] for i in sample_index])

    res = list(itertools.chain.from_iterable(res))
    print(f'Sample {len(res)} documents.')

    # saving cache
    with open(f'{cache_dir}/sample_id.pkl', 'wb') as fout:
        pickle.dump(ids, fout)

    return res

def select_examples_from_question(question, num_examples, clusters_dict):
    "Return a list of examples randomly in question's cluster"
    # !!!important!!! make sure that you called set_seed
    for cluster in clusters_dict.values():
        if next((item for item in cluster if item["question"] == question), None):
            res = list(random.sample(cluster, num_examples))
            return res
    

def main(args):
    if args.is_exist_cluster:
        print("Loading clustering dictionary file ...")
        cluster_dic = load_clusters(args.cluster_path)
    else:
        cluster_dic = create_clusters(args.embed_model_name, args.domain_data_paths, args.min_community_size, args.threshold, args.cluster_path, args.cache_dir, args.force_rebuild, verbose=True)
    sample_questions = select_samples(args.num_sample, cluster_dic, args.cache_dir)
    
    with open(args.sample_question_path, 'w') as fout:
        for ques in sample_questions:
            fout.write(json.dumps(ques, indent=2, default=str)+'\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument('--embed_model_name', '-m', type=str, help="pretrained model for embedding questions",default='sentence-transformers/all-mpnet-base-v2')
    argparser.add_argument("--min_community_size", type=int, default=10)
    argparser.add_argument('--threshold', type=float, default=0.75)
    argparser.add_argument('--num_sample', '-n', type=int, help='the number of questions for each domain math/science/chat', default=5000)
    argparser.add_argument('--verbose', type=bool, default=True)
    argparser.add_argument('--cluster_path', type=str, default='cluster.json')
    argparser.add_argument('--sample_question_path', type=str, help='path to save sampled questions', default='sample_question.jsonl')
    argparser.add_argument('--is_exist_cluster', action='store_false', help='if load from old cluster or create new cluster', default=False)
    
    args = argparser.parse_args()
    # get questions from aggregate of domain data (MATH / SCIENCE / CONVERSATIONAL CHAT)
    # Example: you have a list of name path dataset which each element includes <local_path,name_of_question_column, name_of_answer_column, args, kwargs>

    # MATH set, num_sample=15K
    # python select_data.py -n 30000 --cluster_path data/math_cluster.json --sample_question_path data/math_question.jsonl
    # Result: Sample 22725 documents.
    # setattr(args, 'domain_data_paths',[
    #     ['gsm8k','question', 'answer', ["main"], {"split":"train"}], 
    #     ['math_qa','Problem', 'correct',[], {"split":"train"}], 
    #     ['math-eval/TAL-SCQ5K','problem', 'answer_value', [], {'data_dir':"TAL-SCQ5K-EN","split":"train"}],
    #     ['meta-math/MetaMathQA', 'query', 'response', [], {}]
    # ])
    # setattr(args, 'cache_dir', 'cache/math')
    # setattr(args, 'force_rebuild', {"corpus_embeddings": False, "cluster": False})

    # CNN set, num_sample=5K
    # python select_data.py -n 30000 --cluster_path data/cnn_cluster.json --sample_question_path data/cnn_question.jsonl
    # setattr(args, 'domain_data_paths',[
    #     ['cnn_dailymail', 'article', 'highlights', ['3.0.0'], {"split": 'train'}]
    # ])
    # setattr(args, 'cache_dir', 'cache/cnn')
    # setattr(args, 'force_rebuild', {"corpus_embeddings": False, "cluster": False})

    # Science, num_sample=20K
    # python select_data.py -n 50000 --cluster_path data/science_cluster.json --sample_question_path data/science_question.jsonl --min_community_size 5
    # Result: 44996
    # setattr(args, 'domain_data_paths',[
    #     ['lighteval/mmlu', 'question', 'answer', ['all'], {"split": 'auxiliary_train'}],
    #     ['lighteval/bbq_helm', 'question', 'references', ['all'], {"split": 'train'}],
    #     ['openbookqa', 'question_stem', 'answerKey', ['main'], {'split': 'train'}],
    # ])
    # setattr(args, 'cache_dir', 'cache/science')
    # setattr(args, 'force_rebuild', {"corpus_embeddings": False, "cluster": True})
    
    print(args)
    # Save args file
    with open(os.path.join('args.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=2))

    set_seed(args.seed)

    main(args)

