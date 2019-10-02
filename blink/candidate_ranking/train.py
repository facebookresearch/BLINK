# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import sys
import time
import numpy as np
import pprint
import shutil

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_retrieval.utils
from blink.candidate_ranking.bert_reranking import BertForReranking
import logging
import utils
from evaluate import evaluate_model_on_dataset, evaluate

logger = None


def main(parameters):
    # Read model
    reranker = utils.get_reranker(parameters)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if parameters["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                parameters["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    parameters["train_batch_size"] = (
        parameters["train_batch_size"] // parameters["gradient_accumulation_steps"]
    )
    train_batch_size = parameters["train_batch_size"]
    evaluation_batch_size = parameters["evaluation_batch_size"]
    gradient_accumulation_steps = parameters["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = parameters["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = None
    number_of_samples_per_dataset = {}

    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    time_start = time.time()
    model_output_path = parameters["model_output_path"]

    # Make sure everything is in order with the output directiory
    if os.path.exists(model_output_path) and os.listdir(model_output_path):
        print(
            "Output directory ({}) already exists and is not empty.".format(
                model_output_path
            )
        )
        answer = input("Would you like to empty the existing directory? [Y/N]\n")
        if answer.strip() == "Y":
            print("Deleteing {}...".format(model_output_path))
            shutil.rmtree(model_output_path)
        else:
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(
                    model_output_path
                )
            )

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    utils.write_to_file(
        os.path.join(model_output_path, "training_parameters.txt"), str(parameters)
    )

    logger = utils.get_logger(model_output_path)
    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    ### Load training data
    train_dataset_name = "aida-train"
    train_samples = utils.read_dataset(
        train_dataset_name, parameters["path_to_preprocessed_json_data"]
    )
    train_samples_filtered = utils.filter_samples(train_samples, parameters["top_k"])
    logger.info(
        "Retained {} out of {} samples".format(
            len(train_samples_filtered), len(train_samples)
        )
    )
    number_of_samples_per_dataset[train_dataset_name] = len(train_samples)

    train_data, train_tensor_data = reranker._process_mentions_for_model(
        parameters["context_key"],
        train_samples_filtered,
        tokenizer,
        parameters["max_seq_length"],
        silent=parameters["silent"],
        logger=logger,
        top_k=parameters["top_k"],
        debug=parameters["debug"],
    )
    train_sampler = RandomSampler(train_tensor_data)
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )
    ###

    ### Loading dev data
    dev_dataset_name = "aida-A"
    dev_samples = utils.read_dataset(
        dev_dataset_name, parameters["path_to_preprocessed_json_data"]
    )
    dev_samples_filtered = utils.filter_samples(dev_samples, parameters["top_k"])
    logger.info(
        "Retained {} out of {} samples".format(
            len(dev_samples_filtered), len(dev_samples)
        )
    )
    number_of_samples_per_dataset[dev_dataset_name] = len(dev_samples)

    dev_data, dev_tensor_data = reranker._process_mentions_for_model(
        parameters["context_key"],
        train_samples_filtered,
        tokenizer,
        parameters["max_seq_length"],
        silent=parameters["silent"],
        logger=logger,
        top_k=parameters["top_k"],
        debug=parameters["debug"],
    )
    dev_sampler = SequentialSampler(dev_tensor_data)
    dev_dataloader = DataLoader(
        dev_tensor_data, sampler=dev_sampler, batch_size=evaluation_batch_size
    )
    ###

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_samples_filtered))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Gradient accumulation steps = %d", gradient_accumulation_steps)

    optimizer, scheduler = reranker.get_scheduler_and_optimizer(
        parameters, train_tensor_data, logger
    )

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = parameters["num_train_epochs"]

    model.train()

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, entity_mask = batch
            loss, _ = model(
                input_ids, segment_ids, input_mask, label_ids, entity_mask=entity_mask
            )

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            tr_loss += loss.item()

            if (step + 1) % (
                parameters["print_tr_loss_opt_steps_interval"]
                * parameters["gradient_accumulation_steps"]
            ) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss
                        / (
                            parameters["print_tr_loss_opt_steps_interval"]
                            * gradient_accumulation_steps
                        ),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), parameters["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (
                parameters["dev_evaluation_interval"]
                * gradient_accumulation_steps
                * train_batch_size
            ) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate_model_on_dataset(
                    model,
                    dev_dataloader,
                    dev_dataset_name,
                    device=device,
                    logger=logger,
                    number_of_samples=number_of_samples_per_dataset[dev_dataset_name],
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate_model_on_dataset(
            model,
            dev_dataloader,
            dev_dataset_name,
            device=device,
            logger=logger,
            path_to_file_to_write_results=output_eval_file,
            number_of_samples=number_of_samples_per_dataset[dev_dataset_name],
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    parameters["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    reranker = utils.get_reranker(parameters)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if parameters["evaluate"]:
        parameters["path_to_model"] = model_output_path
        evaluate(parameters, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_preprocessed_json_data",
        default="data/train_and_benchmark_processed_json",
        type=str,
        help="The path to the train and benchmarking data.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-large-cased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--model_output_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where the trained model is to be dumped.",
    )

    parser.add_argument(
        "--context_key", default="tagged_query_context_sent_prev_curr_next", type=str
    )
    parser.add_argument(
        "--lowercase_flag",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument("--top_k", default=80, type=int)
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Whether to run evaluation."
    )
    parser.add_argument(
        "--full_evaluation",
        action="store_true",
        help="Whether to run the evaluation on all datasets.",
    )
    parser.add_argument(
        "--evaluate_with_pregenerated_candidates",
        action="store_true",
        help="Whether to run in debug mode with only 200 samples.",
    )
    parser.add_argument(
        "--output_eval_file",
        default=None,
        type=str,
        help="The txt file where the the evaluation results will be written.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode with only 200 samples.",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Whether to print progress bars."
    )

    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--evaluation_batch_size",
        default=4,
        type=int,
        help="Total batch size for evaluation.",
    )

    parser.add_argument(
        "--dataparallel_bert",
        action="store_true",
        help="Whether to distributed the candidate generation process.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--print_tr_loss_opt_steps_interval",
        type=int,
        default=20,
        help="Interval of loss printing",
    )
    parser.add_argument(
        "--dev_evaluation_interval",
        type=int,
        default=160,
        help="Interval for evaluation during training",
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Interval for model saving"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    parameters = args.__dict__
    main(parameters)
