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
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.biencoder import BiEncoderRanker
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]	
        candidate_input = batch[1]
        mention_idxs = batch[-1]
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input, mention_idxs=mention_idxs)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_samples = utils.read_dataset("train", params["data_path"])
    logger.info("Read %d train samples." % len(train_samples))

    candidate_token_ids = None
    entity2id = None
    tokenized_contexts_dir = ""
    if not params["debug"]:
        candidate_token_ids = torch.load("/private/home/belindali/BLINK/models/entity_token_ids_128.t7") # TODO DONT HARDCODE THESE PATHS
        id2line = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines() # TODO DONT HARDCODE THESE PATHS
        entity2id = {json.loads(id2line[i])['entity']: i for i in range(len(id2line))}
        tokenized_contexts_dir = os.path.join("/private/home/belindali/BLINK/models/tokenized_contexts/", params["data_path"].split('/')[-1]) # TODO DONT HARDCODE THESE PATHS
    train_data, train_tensor_data_tuple = data.process_mention_data(
        samples=train_samples,
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not args.no_mention_bounds),
        candidate_token_ids=candidate_token_ids,
        entity2id=entity2id,
        saved_context_file=os.path.join(tokenized_contexts_dir, "train.json"),
        get_cached_representation=(not params["debug"]),
    )
    logger.info("Finished reading train samples")

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid", params["data_path"])
    valid_subset = 2048
    logger.info("Read %d valid samples, choosing %d subset" % (len(valid_samples), valid_subset))

    valid_data, valid_tensor_data = data.process_mention_data(
        samples=valid_samples[:valid_subset],  # use subset of valid data TODO Make subset random????
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not args.no_mention_bounds),
        candidate_token_ids=candidate_token_ids,
        entity2id=entity2id,
        saved_context_file=os.path.join(tokenized_contexts_dir, "valid.json"),
        get_cached_representation=(not params["debug"]),
    )
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )
    # save memory
    candidate_token_ids = None
    entity2id = None

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger,
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    num_train_epochs = params["num_train_epochs"]
    num_samples_per_batch = len(train_samples) // num_train_epochs

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, min(len(train_tensor_data_tuple[0]), num_samples_per_batch), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1
    logger.info("Num samples per batch : %d" % num_samples_per_batch)
    for epoch_idx in trange(params["last_epoch"], int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        start_idx = epoch_idx * num_samples_per_batch
        end_idx = (epoch_idx + 1) * num_samples_per_batch

        batch_train_tensor_data = TensorDataset(
            *[element[start_idx:end_idx] for element in train_tensor_data_tuple]
        )
        if params["shuffle"]:
            train_sampler = RandomSampler(batch_train_tensor_data)
        else:
            train_sampler = SequentialSampler(batch_train_tensor_data)

        train_dataloader = DataLoader(
            batch_train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0]	
            candidate_input = batch[1]
            mention_idxs = batch[-1]
            loss, _ = reranker(context_input, candidate_input, mention_idxs=mention_idxs)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                model.train()
                logger.info("\n")

                # if (step + 1) % (params["eval_interval"] * grad_acc_steps * 10) == 0:
                #     logger.info("***** Saving fine - tuned model *****")
                #     epoch_output_folder_path = os.path.join(
                #         model_output_path, "epoch_{}_step_{}".format(epoch_idx, step + 1)
                #     )
                #     utils.save_model(model, tokenizer, epoch_output_folder_path)

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
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
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
