"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DESTINATION_PATH = Path("data/synthetic/v2/tokenized/7B")
INPUT_DATA_PATH = Path("data/synthetic/v2/raw_data")
CHECKPOINT_DIR = Path("checkpoints/meta-llama/Llama-2-7b-hf")
IGNORE_INDEX = -1
MASK_INPUTS = False
SEED = 42
MAX_SEQ_LEN = 2048


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    input_data_path: Path = INPUT_DATA_PATH,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
    max_seq_length: Optional[int] = MAX_SEQ_LEN,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and valid dataset saved as `train.pt` and `valid.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading data file...")

    data_train = []
    domains = ['chat', 'cnn', 'math', 'science']
    for domain in domains:
        print(f"Loading {domain} domain ...")
        with open(input_data_path / f'{domain}_prompt_data.jsonl', "r", encoding="utf-8") as fin:
            for line in tqdm(fin):
                data_train.append(json.loads(line))

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and valid
    # breakpoint()
    train_set, _ = random_split(
        data_train, [1.0, 0.0], generator=torch.Generator().manual_seed(seed)
    )
    train_set = list(train_set)

    print(f"train has {len(train_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = example['instruction'] + example['input']
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)