#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import numpy as np
import pickle
from filelock import FileLock
import dill
from datasets import load_from_disk
from datasets import concatenate_datasets
from termcolor import colored


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # Han: many arguments below will not be used, but keeping for future edits
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library). For example, Wikipedia.",
    )
    parser.add_argument(
        "--additional_dataset_name",
        type=str,
        default=None,
        help="The name of the additional dataset to use (via the datasets library). For example, BookCorpus.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--raw_data_percentage",
        default=100,
        help="The percentage of raw data used as the train set",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.0, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--no_save_grads", action="store_true", help="Whether to save gradients to a file.")
    parser.add_argument(
        "--query_file", type=str, default=None, help="A pickle file containing gradient information from the querying data."
    )
    parser.add_argument(
        "--query_data_cap", type=int, default=None, help="Max number of data for which we will save gradients.",
    )
    parser.add_argument("--influence_metric", type=str, default=None, help="Metric for computing the gradients.")
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--tokenized_data_file_path", type=str, default=None, help="Path of the tokenized data file."
    )
    parser.add_argument(
        "--if_create_tokenized_data_file", type=str, default=None, help="Whether to create a new tokenized data file (yes or no)."
    )
    parser.add_argument(
        "--load_trained_score_based_model",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--decode_with_ald",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--sigma_start_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_end_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_num_steps", type=int, default=1000, help="",
    )
    parser.add_argument(
        "--loss_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--remove_noise_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--hardcoded_pseudo_diralpha", type=float, default=3, help="",
    )
    parser.add_argument(
        "--context_size", type=int, default=0, help="",
    )
    parser.add_argument(
        "--decoding_block_size", type=int, default=25, help="",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    assert args.use_slow_tokenizer == True # Han: for a compatible tokenizer with the prompt tuning model
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    jump = False
    if args.tokenized_data_file_path and args.if_create_tokenized_data_file:
        if args.if_create_tokenized_data_file == "no":
            raise ValueError("cannot load existing dataset file in the data processing file")
            tokenized_datasets = load_from_disk(args.tokenized_data_file_path)
            jump = True
        elif args.if_create_tokenized_data_file == "yes":
            if accelerator.is_main_process:
                pass
        else:
            raise ValueError("check args.if_create_tokenized_data_file")

    raw_datasets = load_dataset(
        args.dataset_name,
        data_files={"train": args.train_file },
    )
    # if not jump:
    #     assert args.dataset_name is not None
    #     raw_datasets = datasets.DatasetDict()
    #     if args.dataset_config_name and args.dataset_config_name != "none":
    #         raw_datasets["train"] = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             split=f"train[:{args.raw_data_percentage}%]",
    #         )
    #     else:
    #         raw_datasets["train"] = load_dataset(
    #             args.dataset_name,
    #             split=f"train[:{args.raw_data_percentage}%]",
    #         )
    #     if args.additional_dataset_name and args.additional_dataset_name != "none":
    #         add_dataset = load_dataset(args.additional_dataset_name, split=f"train[:{args.raw_data_percentage}%]")
    #         raw_datasets["train"] = raw_datasets["train"].remove_columns("title")
    #         raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], add_dataset])

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # rather than group and batch out texts, we take an example and split it out into multiple
    # examples, based on the decoding block size
    def split_text(examples):
        srcs, tgts = examples["src"], examples["trg"]
        pairs = []
        for src, tgt in zip(srcs, tgts):
            while len(tgt) > args.decoding_block_size:
                pairs.append({"src": src, "tgt": tgt[:args.decoding_block_size]})
                src += tgt[:args.decoding_block_size]
                tgt = tgt[args.decoding_block_size:]
            # final pair :)
            pairs.append({"src": src, "tgt": tgt[:args.decoding_block_size]})
        return {
            "src": [x["src"] for x in pairs],
            "trg": [x["tgt"] for x in pairs],
        }

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            split_text,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting examples for decoding block size {args.decoding_block_size}",
        )

    # tokenize everything
    def tokenize_function(examples):
        return {
            "src_input_ids": tokenizer(examples["src"]).input_ids[:-1],  # remove eos_token 
            "tgt_input_ids": tokenizer(examples["trg"]).input_ids[1:]  # no bos token
        }

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )

    # drop old columns, rename new ones
    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.remove_columns(["src", "trg"])
        tokenized_datasets = tokenized_datasets.rename_column("src_input_ids", "input_ids")
        tokenized_datasets = tokenized_datasets.rename_column("tgt_input_ids", "labels")

    if args.tokenized_data_file_path and args.if_create_tokenized_data_file:
        if args.if_create_tokenized_data_file == "yes":
            if accelerator.is_main_process:
                tokenized_datasets.save_to_disk(args.tokenized_data_file_path)


if __name__ == "__main__":
    main()
