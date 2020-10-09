# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import faiss
import pickle
import torch
import json
import sys
import io
import random
import time
import traceback
import numpy as np
from scipy.special import softmax, expit

import torch.nn.functional as F

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from elq.biencoder.biencoder import BiEncoderRanker
from elq.vcg_utils.measures import entity_linking_tp_with_overlap
import logging

import elq.candidate_ranking.utils as utils
from elq.biencoder.data_process import process_mention_data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from elq.common.params import ElqParser
from elq.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer, DenseIVFFlatIndexer


logger = None
np.random.seed(1234)  # reproducible for FAISS indexer

def evaluate(
    reranker, eval_dataloader, params, device, logger,
    cand_encs=None, faiss_index=None,
    get_losses=False,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_num_correct = 0.0
    eval_num_p = 0.0
    eval_num_g = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0
    overall_loss = 0.0

    if cand_encs is not None and not params["freeze_cand_enc"]:
        torch.cuda.empty_cache()
        cand_encs = cand_encs.to(device)

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]	
        candidate_input = batch[1]
        # (bs, num_actual_spans)
        label_ids = batch[2].cpu().numpy() if params["freeze_cand_enc"] else None
        if params["debug"] and label_ids is not None:
            label_ids[label_ids > 199] = 199
        mention_idx = batch[-2].cpu().numpy()
        mention_idx_mask = batch[-1].cpu().numpy()
        
        with torch.no_grad():
            # evaluate with joint mention detection
            if params["freeze_cand_enc"]:
                context_outs = reranker.encode_context(
                    context_input,
                    num_cand_mentions=50,
                    topK_threshold=-3.5,
                )
                embedding_context = context_outs['mention_reps'].cpu().numpy()
                pred_mention_mask = context_outs['mention_masks'].cpu().numpy()
                chosen_mention_bounds = context_outs['mention_bounds'].cpu().numpy()
                embedding_ctxt = embedding_context[pred_mention_mask]
                # do faiss search for closest entity
                # DIM (all_pred_mentions_batch, num_cand_entities); (all_pred_mentions_batch, num_cand_entities)
                top_cand_logits_shape, top_cand_indices_shape = faiss_index.search_knn(embedding_ctxt, 10)
                top_cand_logits = np.zeros((pred_mention_mask.shape[0], pred_mention_mask.shape[1], 10), dtype=np.float)
                top_cand_indices = np.zeros_like(pred_mention_mask, dtype=np.int)
                top_cand_logits[pred_mention_mask] = top_cand_logits_shape
                top_cand_indices[pred_mention_mask] = top_cand_indices_shape[:,0]
                scores = (np.log(softmax(top_cand_logits, -1)) + torch.sigmoid(context_outs['mention_logits'].unsqueeze(-1)).log().cpu().numpy())[:,:,0]
                tmp_num_correct = 0.0
                tmp_num_p = 0.0
                tmp_num_g = 0.0
                for i, ex in enumerate(top_cand_indices):
                    gold_mb = mention_idx[i][mention_idx_mask[i]]
                    gold_label_ids = label_ids[i][mention_idx_mask[i]]
                    overall_score_mask = scores[i][pred_mention_mask[i]] > -2.5
                    pred_mb = chosen_mention_bounds[i][pred_mention_mask[i]][overall_score_mask]
                    pred_label_ids = ex[pred_mention_mask[i]][overall_score_mask]
                    gold_triples = [(str(gold_label_ids[j]), gold_mb[j][0], gold_mb[j][1]) for j in range(len(gold_mb))]
                    pred_triples = [(str(pred_label_ids[j]), pred_mb[j][0], pred_mb[j][1]) for j in range(len(pred_mb))]
                    num_overlap_weak, _ = entity_linking_tp_with_overlap(gold_triples, pred_triples)
                    tmp_num_correct += num_overlap_weak
                    tmp_num_p += float(len(pred_triples))
                    tmp_num_g += float(len(gold_triples))
                text_encs = embedding_context
            else:
                loss, logits, mention_logits, mention_bounds = reranker(
                    context_input, candidate_input,
                    cand_encs=cand_encs,
                    gold_mention_bounds=batch[-2],
                    gold_mention_bounds_mask=batch[-1],
                    return_loss=True,
                )
                logits = logits.cpu().numpy()
                # Using in-batch negatives, the label ids are diagonal
                label_ids = torch.LongTensor(torch.arange(logits.shape[0]))
                label_ids = label_ids.cpu().numpy()
                tmp_num_correct = utils.accuracy(logits, label_ids)
                tmp_num_p = len(batch[-2][batch[-1]])
                tmp_num_g = len(batch[-2][batch[-1]])

                overall_loss += loss

        eval_num_correct += tmp_num_correct
        eval_num_p += tmp_num_p
        eval_num_g += tmp_num_g

        nb_eval_steps += 1

    if cand_encs is not None:
        cand_encs = cand_encs.to("cpu")
        torch.cuda.empty_cache()

    if nb_eval_steps > 0 and overall_loss > 0:
        normalized_overall_loss = overall_loss / nb_eval_steps
        logger.info("Overall loss: %.5f" % normalized_overall_loss)
    if eval_num_p > 0:
        normalized_eval_p = eval_num_correct / eval_num_p
    else:
        normalized_eval_p = 0.0
    if eval_num_g > 0:
        normalized_eval_r = eval_num_correct / eval_num_g
    else:
        normalized_eval_r = 0.0
    logger.info("Precision: %.5f" % normalized_eval_p)
    logger.info("Recall: %.5f" % normalized_eval_r)
    if normalized_eval_p + normalized_eval_r == 0:
        f1 = 0
    else:
        f1 = 2 * normalized_eval_p * normalized_eval_r / (normalized_eval_p + normalized_eval_r)
    logger.info("F1: %.5f" % f1)
    results["normalized_f1"] = f1
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

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
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
    logger.info("Finished reading all train samples")

    # Load eval data
    try:
        valid_samples = utils.read_dataset("valid", params["data_path"])
    except FileNotFoundError:
        valid_samples = utils.read_dataset("dev", params["data_path"])
    # MUST BE DIVISBLE BY n_gpus
    if len(valid_samples) > 1024:
        valid_subset = 1024
    else:
        valid_subset = len(valid_samples) - len(valid_samples) % torch.cuda.device_count()
    logger.info("Read %d valid samples, choosing %d subset" % (len(valid_samples), valid_subset))

    valid_data, valid_tensor_data, extra_ret_values = process_mention_data(
        samples=valid_samples[:valid_subset],  # use subset of valid data
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        context_key=params["context_key"],
        title_key=params["title_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not args.no_mention_bounds),
        candidate_token_ids=None,
        params=params,
    )
    candidate_token_ids = extra_ret_values["candidate_token_ids"]
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # load candidate encodings
    cand_encs = None
    cand_encs_index = None
    if params["freeze_cand_enc"]:
        cand_encs = torch.load(params['cand_enc_path'])
        logger.info("Loaded saved entity encodings")
        if params["debug"]:
            cand_encs = cand_encs[:200]
        
        # build FAISS index
        cand_encs_index = DenseHNSWFlatIndexer(1)
        cand_encs_index.deserialize_from(params['index_path'])
        logger.info("Loaded FAISS index on entity encodings")
        num_neighbors = 10

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params,
        cand_encs=cand_encs, device=device,
        logger=logger, faiss_index=cand_encs_index,
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
    if params["dont_distribute_train_samples"]:
        num_samples_per_batch = len(train_samples)

        train_data, train_tensor_data_tuple, extra_ret_values = process_mention_data(
            samples=train_samples,
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            max_cand_length=params["max_cand_length"],
            context_key=params["context_key"],
            title_key=params["title_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            add_mention_bounds=(not args.no_mention_bounds),
            candidate_token_ids=candidate_token_ids,
            params=params,
        )
        logger.info("Finished preparing training data")
    else:
        num_samples_per_batch = len(train_samples) // num_train_epochs


    trainer_path = params.get("path_to_trainer_state", None)
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(
        params, optimizer, num_samples_per_batch,
        logger
    )
    if trainer_path is not None and os.path.exists(trainer_path):
        training_state = torch.load(trainer_path)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info("Loaded saved training state")

    model.train()

    best_epoch_idx = -1
    best_score = -1
    logger.info("Num samples per batch : %d" % num_samples_per_batch)
    for epoch_idx in trange(params["last_epoch"] + 1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if not params["dont_distribute_train_samples"]:
            start_idx = epoch_idx * num_samples_per_batch
            end_idx = (epoch_idx + 1) * num_samples_per_batch

            train_data, train_tensor_data_tuple, extra_ret_values = process_mention_data(
                samples=train_samples[start_idx:end_idx],
                tokenizer=tokenizer,
                max_context_length=params["max_context_length"],
                max_cand_length=params["max_cand_length"],
                context_key=params["context_key"],
                title_key=params["title_key"],
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                add_mention_bounds=(not args.no_mention_bounds),
                candidate_token_ids=candidate_token_ids,
                params=params,
            )
            logger.info("Finished preparing training data for epoch {}: {} samples".format(epoch_idx, len(train_tensor_data_tuple[0])))
    
        batch_train_tensor_data = TensorDataset(
            *list(train_tensor_data_tuple)
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
            label_ids = batch[2] if params["freeze_cand_enc"] else None
            mention_idxs = batch[-2]
            mention_idx_mask = batch[-1]
            if params["debug"] and label_ids is not None:
                label_ids[label_ids > 199] = 199

            cand_encs_input = None
            label_input = None
            mention_reps_input = None
            mention_logits = None
            mention_bounds = None
            hard_negs_mask = None
            if params["adversarial_training"]:
                assert cand_encs is not None and label_ids is not None  # due to params["freeze_cand_enc"] being set
                '''
                GET CLOSEST N CANDIDATES (AND APPROPRIATE LABELS)
                '''
                # (bs, num_spans, embed_size)
                pos_cand_encs_input = cand_encs[label_ids.to("cpu")]
                pos_cand_encs_input[~mention_idx_mask] = 0

                context_outs = reranker.encode_context(
                    context_input, gold_mention_bounds=mention_idxs,
                    gold_mention_bounds_mask=mention_idx_mask,
                    get_mention_scores=True,
                )
                mention_logits = context_outs['all_mention_logits']
                mention_bounds = context_outs['all_mention_bounds']
                mention_reps = context_outs['mention_reps']
                # mention_reps: (bs, max_num_spans, embed_size) -> masked_mention_reps: (all_pred_mentions_batch, embed_size)
                masked_mention_reps = mention_reps[context_outs['mention_masks']]

                # neg_cand_encs_input_idxs: (all_pred_mentions_batch, num_negatives)
                _, neg_cand_encs_input_idxs = cand_encs_index.search_knn(masked_mention_reps.detach().cpu().numpy(), num_neighbors)
                neg_cand_encs_input_idxs = torch.from_numpy(neg_cand_encs_input_idxs)
                # set "correct" closest entities to -1
                # masked_label_ids: (all_pred_mentions_batch)
                masked_label_ids = label_ids[mention_idx_mask]
                # neg_cand_encs_input_idxs: (max_spans_in_batch, num_negatives)
                neg_cand_encs_input_idxs[neg_cand_encs_input_idxs - masked_label_ids.to("cpu").unsqueeze(-1) == 0] = -1

                # reshape back tensor (extract num_spans dimension)
                # (bs, num_spans, num_negatives)
                neg_cand_encs_input_idxs_reconstruct = torch.zeros(label_ids.size(0), label_ids.size(1), neg_cand_encs_input_idxs.size(-1), dtype=neg_cand_encs_input_idxs.dtype)
                neg_cand_encs_input_idxs_reconstruct[mention_idx_mask] = neg_cand_encs_input_idxs
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs_reconstruct

                # create neg_example_idx (corresponding example (in batch) for each negative)
                # neg_example_idx: (bs * num_negatives)
                neg_example_idx = torch.arange(neg_cand_encs_input_idxs.size(0)).unsqueeze(-1)
                neg_example_idx = neg_example_idx.expand(neg_cand_encs_input_idxs.size(0), neg_cand_encs_input_idxs.size(2))
                neg_example_idx = neg_example_idx.flatten()

                # flatten and filter -1 (i.e. any correct/positive entities)
                # neg_cand_encs_input_idxs: (bs * num_negatives, num_spans)
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs.permute(0,2,1)
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs.reshape(-1, neg_cand_encs_input_idxs.size(-1))
                # mask invalid negatives (actually the positive example)
                # (bs * num_negatives)
                mask = ~((neg_cand_encs_input_idxs == -1).sum(1).bool())  # rows without any -1 entry
                # deletes corresponding negative for *all* spans in that example (deletes at most 3 of 10 negatives / example)
                # neg_cand_encs_input_idxs: (bs * num_negatives - invalid_negs, num_spans)
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs[mask]
                # neg_cand_encs_input_idxs: (bs * num_negatives - invalid_negs)
                neg_example_idx = neg_example_idx[mask]
                # (bs * num_negatives - invalid_negs, num_spans, embed_size)
                neg_cand_encs_input = cand_encs[neg_cand_encs_input_idxs]
                # (bs * num_negatives - invalid_negs, num_spans, embed_size)
                neg_mention_idx_mask = mention_idx_mask[neg_example_idx]
                neg_cand_encs_input[~neg_mention_idx_mask] = 0

                # create input tensors (concat [pos examples, neg examples])
                # (bs + bs * num_negatives, num_spans, embed_size)
                mention_reps_input = torch.cat([
                    mention_reps, mention_reps[neg_example_idx.to(device)],
                ])
                assert mention_reps.size(0) == pos_cand_encs_input.size(0)

                # (bs + bs * num_negatives, num_spans)
                label_input = torch.cat([
                    torch.ones(pos_cand_encs_input.size(0), pos_cand_encs_input.size(1), dtype=label_ids.dtype),
                    torch.zeros(neg_cand_encs_input.size(0), neg_cand_encs_input.size(1), dtype=label_ids.dtype),
                ]).to(device)
                # (bs + bs * num_negatives, num_spans, embed_size)
                cand_encs_input = torch.cat([
                    pos_cand_encs_input, neg_cand_encs_input,
                ]).to(device)
                hard_negs_mask = torch.cat([mention_idx_mask, neg_mention_idx_mask])

            loss, _, _, _ = reranker(
                context_input, candidate_input,
                cand_encs=cand_encs_input, text_encs=mention_reps_input,
                mention_logits=mention_logits, mention_bounds=mention_bounds,
                label_input=label_input, gold_mention_bounds=mention_idxs,
                gold_mention_bounds_mask=mention_idx_mask,
                hard_negs_mask=hard_negs_mask,
                return_loss=True,
            )

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
                loss = None  # for GPU mem management
                mention_reps = None
                mention_reps_input = None
                label_input = None
                cand_encs_input = None

                evaluate(
                    reranker, valid_dataloader, params,
                    cand_encs=cand_encs, device=device,
                    logger=logger, faiss_index=cand_encs_index,
                    get_losses=params["get_losses"],
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, os.path.join(epoch_output_folder_path, "training_state.th"))

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        logger.info("Valid data evaluation")
        results = evaluate(
            reranker, valid_dataloader, params,
            cand_encs=cand_encs, device=device,
            logger=logger, faiss_index=cand_encs_index,
            get_losses=params["get_losses"],
        )
        logger.info("Train data evaluation")
        results = evaluate(
            reranker, train_dataloader, params,
            cand_encs=cand_encs, device=device,
            logger=logger, faiss_index=cand_encs_index,
            get_losses=params["get_losses"],
        )

        ls = [best_score, results["normalized_f1"]]
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
        evaluate(params, cand_encs=cand_encs, logger=logger, faiss_index=cand_encs_index)


if __name__ == "__main__":
    parser = ElqParser(add_model_args=True)
    parser.add_training_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
