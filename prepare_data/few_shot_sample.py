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
            fout.write(json.dumps(ques, indent=2, default=str)+'\n')

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
        query_embedding = corpus_embeddings[sample_id]
        # query_embedding = model.encode(queries, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
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

    domain = 'complexqa1' # math, science, cnn, complexqa, bigbench, complexqa1


    subset = ['abstract_narrative_understanding', 'anachronisms', 'analogical_similarity', 'analytic_entailment', 'arithmetic', 'ascii_word_recognition', 'authorship_verification', 'auto_categorization', 'auto_debugging', 'bbq_lite_json', 'bridging_anaphora_resolution_barqa', 'causal_judgment', 'cause_and_effect', 'checkmate_in_one', 'chess_state_tracking', 'chinese_remainder_theorem', 'cifar10_classification', 'code_line_description', 'codenames', 'color', 'common_morpheme', 'conceptual_combinations', 'conlang_translation', 'contextual_parametric_knowledge_conflicts', 'crash_blossom', 'crass_ai', 'cryobiology_spanish', 'cryptonite', 'cs_algorithms', 'dark_humor_detection', 'date_understanding', 'disambiguation_qa', 'discourse_marker_prediction', 'disfl_qa', 'dyck_languages', 'elementary_math_qa', 'emoji_movie', 'emojis_emotion_prediction', 'empirical_judgments', 'english_proverbs', 'english_russian_proverbs', 'entailed_polarity', 'entailed_polarity_hindi', 'epistemic_reasoning', 'evaluating_information_essentiality', 'fact_checker', 'fantasy_reasoning', 'few_shot_nlg', 'figure_of_speech_detection', 'formal_fallacies_syllogisms_negation', 'gem', 'gender_inclusive_sentences_german', 'general_knowledge', 'geometric_shapes', 'goal_step_wikihow', 'gre_reading_comprehension', 'hhh_alignment', 'hindi_question_answering', 'hindu_knowledge', 'hinglish_toxicity', 'human_organs_senses', 'hyperbaton', 'identify_math_theorems', 'identify_odd_metaphor', 'implicatures', 'implicit_relations', 'indic_cause_and_effect', 'intent_recognition', 'international_phonetic_alphabet_nli', 'international_phonetic_alphabet_transliterate', 'intersect_geometry', 'irony_identification', 'kanji_ascii', 'kannada', 'key_value_maps', 'known_unknowns', 'language_games', 'language_identification', 'linguistic_mappings', 'linguistics_puzzles', 'list_functions', 'logic_grid_puzzle', 'logical_args', 'logical_deduction', 'logical_fallacy_detection', 'logical_sequence', 'mathematical_induction', 'matrixshapes', 'medical_questions_russian', 'metaphor_boolean', 'metaphor_understanding', 'minute_mysteries_qa', 'misconceptions', 'misconceptions_russian', 'mnist_ascii', 'modified_arithmetic', 'moral_permissibility', 'movie_dialog_same_or_different', 'movie_recommendation', 'mult_data_wrangling', 'navigate', 'nonsense_words_grammar', 'novel_concepts', 'object_counting', 'odd_one_out', 'operators', 'paragraph_segmentation', 'parsinlu_qa', 'parsinlu_reading_comprehension', 'penguins_in_a_table', 'periodic_elements', 'persian_idioms', 'phrase_relatedness', 'physical_intuition', 'physics', 'physics_questions', 'play_dialog_same_or_different', 'polish_sequence_labeling', 'presuppositions_as_nli', 'qa_wikidata', 'question_selection', 'real_or_fake_text', 'reasoning_about_colored_objects', 'repeat_copy_logic', 'rephrase', 'rhyming', 'riddle_sense', 'ruin_names', 'salient_translation_error_detection', 'scientific_press_release', 'semantic_parsing_in_context_sparc', 'semantic_parsing_spider', 'sentence_ambiguity', 'similarities_abstraction', 'simp_turing_concept', 'simple_arithmetic_json', 'simple_arithmetic_json_subtasks', 'simple_ethical_questions', 'simple_text_editing', 'snarks', 'social_iqa', 'social_support', 'sports_understanding', 'strange_stories', 'strategyqa', 'sufficient_information', 'suicide_risk', 'swahili_english_proverbs', 'swedish_to_german_proverbs', 'symbol_interpretation', 'tellmewhy', 'temporal_sequences', 'tense', 'timedial', 'topical_chat', 'tracking_shuffled_objects', 'understanding_fables', 'undo_permutation', 'unit_conversion', 'unit_interpretation', 'unnatural_in_context_learning', 'vitaminc_fact_verification', 'what_is_the_tao', 'which_wiki_edit', 'winowhy', 'word_sorting', 'word_unscrambling']
    
    multi_domain_data_paths = {
        "math": [
            ['gsm8k','question', 'answer', ["main"], {"split":"train"}], 
            ['math_qa','Problem', 'correct',[], {"split":"train"}], 
            ['math-eval/TAL-SCQ5K','problem', 'answer_value', [], {'data_dir':"TAL-SCQ5K-EN","split":"train"}],
            ['meta-math/MetaMathQA', 'query', 'response', [], {}],
            ['TIGER-Lab/MathInstruct', 'instruction', 'output', [], {"split":"train"}]
        ],
        "bigbench": [['tasksource/bigbench','inputs', 'targets', [set], {"split":"train"}] for set in subset],

        "complexqa1": [
            ['medmcqa', 'question', 'cop', [], {"split": 'train'}],
            ['winogrande', 'sentence', 'answer', ['winogrande_xl'], {"split": 'train'}],
            ['winogrande', 'sentence', 'answer', ['winogrande_debiased'], {"split": 'train'}],
            ['boolq', 'question', 'answer', [], {"split": 'train'}],
            ['sciq', 'question', 'correct_answer', [], {"split": 'train'}]
        ],

        "cnn": [
            ['cnn_dailymail', 'article', 'highlights', ['3.0.0'], {"split": 'train'}]
        ],

        "science": [
            ['lighteval/mmlu', 'question', 'answer', ['all'], {"split": 'auxiliary_train'}],
            ['lighteval/bbq_helm', 'question', 'references', ['all'], {"split": 'train'}],
            ['openbookqa', 'question_stem', 'answerKey', ['main'], {'split': 'train'}],
        ],

        "complexqa": [
            ['ai2_arc', 'question', 'answerKey', ['ARC-Challenge'], {'split': 'train'}],
            ['ai2_arc', 'question', 'answerKey', ['ARC-Easy'], {'split': 'train'}],
            ['piqa', 'goal', 'label', [], {'split': 'train'}],
            ['social_i_qa', 'question', 'label', [], {'split': 'train'}],
            ['Muennighoff/babi', 'question', 'answer', [], {'split': 'train'}],
            ['Rowan/hellaswag', 'ctx', 'label', [], {'split': 'train'}],
        ]
    }

    print(f'Loading {domain} sample id')
    cache_dir = f'cache/{domain}'
    with open(f'{cache_dir}/sample_id.pkl', 'rb') as fin:
        sample_id = pickle.load(fin)

    print('Creating data')
    domain_data_paths = multi_domain_data_paths[domain]
    create_few_shot(model, domain_data_paths, sample_id, f'data/few-shot/{domain}_fewshot.jsonl', cache_dir=cache_dir)

