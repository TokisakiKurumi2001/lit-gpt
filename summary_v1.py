import os, json
from typing import List
from dataclasses import dataclass
import argparse

@dataclass
class RawDataInstance:
    path: str
    stats: List[str]

def avg(a: List[float]):
    return sum(a) / len(a)

class Parser:
    def __init__(self, version: str):
        self.mappings = {
            "mmlu": ["exact_match", "exact_match+fairness", "exact_match+robustness", "inference_runtime"],
            "bbq": ["exact_match", "inference_runtime"],
            "truthful_qa": ["exact_match", "exact_match+fairness", "exact_match+robustness", "inference_runtime"],
            "summarization_cnndm": [
                "rouge_2",
                "bias_metric:mode=representation,demographic_category=gender", # CNN/DailyMail - Representation (gender)
                "bias_metric:mode=representation,demographic_category=race", # CNN/DailyMail - Representation (race)
                "bias_metric:mode=associations,demographic_category=gender,target_category=profession", # CNN/DailyMail - Stereotypes (gender)
                "bias_metric:mode=associations,demographic_category=race,target_category=profession", # CNN/DailyMail - Stereotypes (race)
            ],
        }
        self.raw_data = []
        self.data = [{'version': version}]

    def show_leaderboard(self):
        json_data = self.data[1:]
        parsed_data = {}
        for data in json_data:
            parsed_data[data['name']] = {}
            for k, v in data['report'].items():
                parsed_data[data['name']][k] = v
        template = f"""
        MMLU - EM                                  {parsed_data['mmlu']['exact_match']}
        CNN/DailyMail - ROUGE-2                    {parsed_data['summarization_cnndm']['rouge_2']}
        TruthfulQA - EM                            {parsed_data['truthful_qa']['exact_match']}
        BBQ - EM                                   {parsed_data['bbq']['exact_match']}
        MMLU - EM (Robustness)                     {parsed_data['mmlu']['exact_match+robustness']}
        TruthfulQA - EM (Robustness)               {parsed_data['truthful_qa']['exact_match+robustness']}
        MMLU - EM (Fairness)                       {parsed_data['mmlu']['exact_match+fairness']}
        TruthfulQA - EM (Fairness)                 {parsed_data['truthful_qa']['exact_match+fairness']}
        CNN/DailyMail - Stereotypes (race)         {parsed_data['summarization_cnndm']["bias_metric:mode=associations,demographic_category=race,target_category=profession"]}
        CNN/DailyMail - Stereotypes (gender)       {parsed_data['summarization_cnndm']["bias_metric:mode=associations,demographic_category=gender,target_category=profession"]}
        CNN/DailyMail - Representation (race)      {parsed_data['summarization_cnndm']["bias_metric:mode=representation,demographic_category=race"]}
        CNN/DailyMail - Representation (gender)    {parsed_data['summarization_cnndm']["bias_metric:mode=representation,demographic_category=gender"]}
        """
        print(template)

    def export_json(self):
        self.parse_raw_data()
        return self.data

    def parse_raw_data(self):
        # Multiple subject/sub-task
        mmlu_data = {'subject': []}
        for metric in self.mappings['mmlu']:
            mmlu_data[metric] = []

        ptr_maps = {'mmlu': mmlu_data}
        keyword_split = {'mmlu': 'subject'}

        for el in self.raw_data:
            spec = el.path.split('/')[-1]
            task_name, conf = spec.split(':')
            if task_name in keyword_split.keys():
                ptr = ptr_maps[task_name]
                # extract subject
                subject = conf.split(',')[0]
                subject_name = subject.split('=')[1]
                ptr[keyword_split[task_name]].append(subject_name)
                for metric_name, metric_value in zip(self.mappings[task_name], el.stats):
                    ptr[metric_name].append(round(metric_value, 2))
            else:
                json_template = {'name': "", "report": {}}
                json_template['name'] = task_name
                for metric_name, metric_value in zip(self.mappings[task_name], el.stats):
                    json_template['report'][metric_name] = round(metric_value, 2)
                self.data.append(json_template)
        
        # For MMLU, Bigbench
        for task_name, data in ptr_maps.items():
            data['name'] = task_name
            data['report'] = {}
            for metric_name in self.mappings[task_name]:
                data['report'][metric_name] = round(avg(data[metric_name]), 2)
            self.data.append(data)

    def name2info(self, path: str):
        task_name = path.split(":")[0]
        metrics = self.mappings[task_name]
        return task_name, metrics

    def extract_data(self, name: str, path: str, metrics: List[str]):
        if name in ['mmlu', 'bbq', 'summarization_cnndm']:
            stats = self.__extract_data_test(path, metrics)
        elif name in ['truthful_qa']:
            stats = self.__extract_data_valid(path, metrics)
        raw_data = RawDataInstance(path, stats)
        self.raw_data.append(raw_data)
        return stats

    def __extract_data_test(self, path: str, metrics: List[str]):
        with open(f'{path}/stats.json', 'r') as f:
            d = json.load(f)
        stats = []
        
        for metric in metrics:
            _stats = []
            perturbation = False
            if "+" in metric:
                perturbation = True
                metric, perturbation_criteria = metric.split("+")
            for item in d: 
                if perturbation:
                    if item['name']['name'] == metric \
                        and item['name']['split'] == 'test' \
                        and 'perturbation' in item['name'] \
                        and item['name']['perturbation']['name'] == perturbation_criteria:
                            _stats.append(item['mean'])
                else:
                    if item['name']['name'] == metric \
                        and item['name']['split'] == 'test' \
                        and 'perturbation' not in item['name']:
                        _stats.append(item['mean'])

            if len(_stats) > 0:
                stats.append(_stats[0])
        return stats

    def __extract_data_valid(self, path: str, metrics: List[str]):
        with open(f'{path}/stats.json', 'r') as f:
            d = json.load(f)
        stats = []
        
        for metric in metrics:
            _stats = []
            perturbation = False
            if "+" in metric:
                perturbation = True
                metric, perturbation_criteria = metric.split("+")
            for item in d: 
                if perturbation:
                    if item['name']['name'] == metric \
                        and item['name']['split'] == 'valid' \
                        and 'perturbation' in item['name'] \
                        and item['name']['perturbation']['name'] == perturbation_criteria:
                            _stats.append(item['mean'])
                else:
                    if item['name']['name'] == metric \
                        and item['name']['split'] == 'valid' \
                        and 'perturbation' not in item['name']:
                        _stats.append(item['mean'])

            if len(_stats) > 0:
                stats.append(_stats[0])
        return stats

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description='Version to run')
    cli_parser.add_argument('--version', help='Determine the input directory/version to retrieve statistical information.')
    args = cli_parser.parse_args()

    dir = 'benchmark_output/runs'
    version = args.version
    parser = Parser(version)
    ls = [i for i in os.listdir(f"{dir}/{version}") if os.path.isdir(f"{dir}/{version}/{i}")]
    ignore_dirs = ['groups', 'eval_cache']
    for directory in ls:
        if directory in ignore_dirs:
            continue
        # if directory.startswith('summarization_cnndm'):
        task_name, metrics = parser.name2info(directory)
        stats = parser.extract_data(task_name, f"{dir}/{version}/{directory}", metrics)
    json_data = parser.export_json()
    with open(f'summary-{version}.json', 'w+') as f:
        json.dump(json_data, f)
    parser.show_leaderboard()