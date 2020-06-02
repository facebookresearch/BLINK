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
import numpy as np

import torch.nn.functional as F

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
np.random.seed(1234)  # reproducible for FAISS indexer

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
    cand_encs=None, faiss_index=None, joint_mention_detection=True,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    eval_num_p = 0.0
    eval_num_r = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    if cand_encs is not None and not params["freeze_cand_enc"]:
        torch.cuda.empty_cache()
        cand_encs = cand_encs.to(device)

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]	
        candidate_input = batch[1]
        # (bs, num_actual_spans)
        label_ids = batch[2] if params["freeze_cand_enc"] else None
        if params["debug"] and label_ids is not None:
            label_ids[label_ids > 199] = 199
        
        with torch.no_grad():
            # evaluate with joint mention detection
            if ((
                hasattr(reranker.model, 'do_mention_detection') and reranker.model.do_mention_detection
            ) or (
                hasattr(reranker.model, 'module') and reranker.model.module.do_mention_detection
            )) and joint_mention_detection:
                mention_idxs = None
            else:
                mention_idxs = batch[-2]
            mention_idx_mask = batch[-1]

            if params["freeze_cand_enc"]:
                # get mention encoding
                embedding_context, mention_logits, mention_bounds = reranker.encode_context(
                    context_input, gold_mention_idxs=mention_idxs, #topK_mention=1,
                    topK_threshold=0.5,
                )
                if embedding_context.size(0) > 0:
                    if mention_idxs is None:
                        # (bs, num_mentions)
                        pred_mention_idx_mask = torch.sigmoid(mention_logits) > 0.5
                    else:
                        # (bs, 2)
                        pred_mention_idx_mask = mention_idx_mask
                    flattened_embedding_contexts = embedding_context[pred_mention_idx_mask]
                    # do faiss search for closest entity
                    D, I = faiss_index.search(flattened_embedding_contexts.contiguous().detach().cpu().numpy(), 1)
                    I = I.flatten()
                    I_reshape = -np.ones(embedding_context.shape[:2], dtype=I.dtype)
                    try:
                        I_reshape[pred_mention_idx_mask.contiguous().detach().cpu().numpy()] = I
                    except:
                        import pdb
                        pdb.set_trace()
                else:
                    I = np.array([])
                    I_reshape = np.array([])
                tmp_eval_accuracy = 0.0
                tmp_num_p = 0.0
                tmp_num_r = 0.0
                for i, ex in enumerate(I_reshape):
                    ex_label_ids = label_ids[i][mention_idx_mask[i]]
                    ex = ex[pred_mention_idx_mask[i].contiguous().detach().cpu().numpy()]
                    # unique-ify ex
                    seen_ex = {}
                    for j in ex:
                        if j not in seen_ex:
                            tmp_eval_accuracy += j in ex_label_ids  # only 1, so +1 if present, -1 if not present
                            seen_ex[j] = 0
                    tmp_num_p += float(len(ex))
                tmp_num_r += float(mention_idx_mask.sum())
                # reranker.tokenizer.decode(context_input[0].tolist())
            else:
                import pdb
                pdb.set_trace()
                logits, mention_logits, mention_bounds = reranker(
                    context_input, candidate_input,
                    cand_encs=cand_encs,# label_input=label_ids,
                    gold_mention_idxs=mention_idxs,
                    gold_mention_idx_mask=mention_idx_mask,
                    return_loss=False,
                )

                logits = logits.detach().cpu().numpy()
                # Using in-batch negatives, the label ids are diagonal
                label_ids = torch.LongTensor(
                    torch.arange(params["eval_batch_size"])
                ).unsqueeze(-1)
                label_ids = label_ids.detach().cpu().numpy()
                tmp_eval_accuracy = utils.accuracy(logits, label_ids)
                tmp_num_p = 0.0
                tmp_num_r = 0.0

        eval_accuracy += tmp_eval_accuracy
        eval_num_p += tmp_num_p
        eval_num_r += tmp_num_r

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    if cand_encs is not None:
        cand_encs = cand_encs.to("cpu")
        torch.cuda.empty_cache()

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if eval_num_p > 0:
        normalized_eval_p = eval_accuracy / eval_num_p
    else:
        normalized_eval_p = 0.0
    if eval_num_r > 0:
        normalized_eval_r = eval_accuracy / eval_num_r
    else:
        normalized_eval_r = 0.0
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    logger.info("Precision: %.5f" % normalized_eval_p)
    logger.info("Recall: %.5f" % normalized_eval_r)
    if normalized_eval_p + normalized_eval_r == 0:
        f1 = 0
    else:
        f1 = 2 * normalized_eval_p * normalized_eval_r / (normalized_eval_p + normalized_eval_r)
    logger.info("F1: %.5f" % f1)
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

    cand_encs = None
    if params["freeze_cand_enc"]:
        cand_encs = torch.load("/private/home/belindali/BLINK/models/all_entities_large.t7")  # TODO DONT HARDCODE THESE PATHS
        cand_encs_npy = np.load("/private/home/belindali/BLINK/models/all_entities_large.npy")  # TODO DONT HARDCODE THESE PATHS
        logger.info("Loaded saved entity encodings")
        if params["debug"]:
            cand_encs = cand_encs[:200]
            cand_encs_npy = cand_encs_npy[:200]
        
        # build FAISS index
        d = cand_encs_npy.shape[1]
        nsplits = 100
        cand_encs_flat_index = faiss.IndexFlatIP(d)
        cand_encs_quantizer = faiss.IndexFlatIP(d)
        assert cand_encs_quantizer.is_trained
        cand_encs_index = faiss.IndexIVFFlat(cand_encs_quantizer, d, nsplits, faiss.METRIC_INNER_PRODUCT)
        assert not cand_encs_index.is_trained
        cand_encs_index.train(cand_encs_npy)  # 15s
        assert cand_encs_index.is_trained
        cand_encs_index.add(cand_encs_npy)  # 41s
        cand_encs_flat_index.add(cand_encs_npy)
        assert cand_encs_index.ntotal == cand_encs_npy.shape[0]
        assert cand_encs_flat_index.ntotal == cand_encs_npy.shape[0]
        cand_encs_index.nprobe = 20
        logger.info("Built and trained FAISS index on entity encodings")
        num_neighbors = 10

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
        get_cached_representation=(not params["debug"] and not params["no_cached_representation"]),
    )
    logger.info("Finished reading train samples")

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid", params["data_path"])
    valid_subset = 1024
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
        get_cached_representation=(not params["debug"] and not params["no_cached_representation"]),
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
        reranker, valid_dataloader, params,
        cand_encs=cand_encs, device=device,
        logger=logger, faiss_index=cand_encs_flat_index,
        joint_mention_detection=True,
    )
    logger.info("Non-end2end")
    results = evaluate(
        reranker, valid_dataloader, params,
        cand_encs=cand_encs, device=device,
        logger=logger, faiss_index=cand_encs_flat_index,
        joint_mention_detection=False,
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
    else:
        num_samples_per_batch = len(train_samples) // num_train_epochs


    trainer_path = params.get("path_to_trainer_state", None)
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, min(
        len(train_tensor_data_tuple[0]), num_samples_per_batch,
    ), logger)
    if trainer_path is not None:
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

        if params["dont_distribute_train_samples"]:
            start_idx = 0
            end_idx = num_samples_per_batch
        else:
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
            label_ids = batch[2] if params["freeze_cand_enc"] else None
            mention_idxs = batch[-2]
            mention_idx_mask = batch[-1]
            if params["debug"] and label_ids is not None:
                label_ids[label_ids > 199] = 199
            # TODO pass in all candidate encodings, AND label_input
            #

            cand_encs_input = None
            label_input = None
            mention_reps_input = None
            mention_logits = None
            mention_bounds = None
            if params["adversarial_training"]:
                cand_encs_index.nprobe = 20
                assert cand_encs is not None and label_ids is not None  # due to params["freeze_cand_enc"] being set
                # TODO GET CLOSEST N CANDIDATES HERE (AND APPROPRIATE LABELS)...
                # (bs, num_spans, embed_size)
                pos_cand_encs_input = cand_encs[label_ids.to("cpu")]
                pos_cand_encs_input[~mention_idx_mask] = 0
                # # reshape back tensors (extract num_spans dimension)
                # # (bs, num_spans, embed_size)
                # pos_cand_encs_input_reconstruct = torch.zeros(label_ids.size(0), label_ids.size(1), pos_cand_encs_input.size(-1), dtype=pos_cand_encs_input.dtype)
                # pos_cand_encs_input_reconstruct[mention_idx_mask] = pos_cand_encs_input
                # pos_cand_encs_input = pos_cand_encs_input_reconstruct

                mention_reps, mention_logits, mention_bounds = reranker.encode_context(
                    context_input, gold_mention_idxs=mention_idxs,
                )
                # mention_reps: (bs, max_num_spans, embed_size) -> masked_mention_reps: (bs * num_spans [masked], embed_size)
                masked_mention_reps = mention_reps.reshape(-1, mention_reps.size(2))[mention_idx_mask.flatten()]

                # neg_cand_encs_input_idxs: (bs * num_spans [masked], num_negatives)
                _, neg_cand_encs_input_idxs = cand_encs_index.search(masked_mention_reps.detach().cpu().numpy(), num_neighbors)
                neg_cand_encs_input_idxs = torch.from_numpy(neg_cand_encs_input_idxs)
                # set "correct" closest entities to -1
                # masked_label_ids: (bs * num_spans [masked])
                masked_label_ids = label_ids.flatten()[mention_idx_mask.flatten()]
                # neg_cand_encs_input_idxs: (bs * num_spans [masked], num_negatives)
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
                mention_idx_mask = torch.cat([mention_idx_mask, neg_mention_idx_mask])
            
            import pdb
            pdb.set_trace()
            loss, _ = reranker(
                context_input, candidate_input,
                cand_encs=cand_encs_input, text_encs=mention_reps_input,
                mention_logits=mention_logits, mention_bounds=mention_bounds,
                label_input=label_input, gold_mention_idxs=mention_idxs,
                gold_mention_idx_mask=mention_idx_mask,
                all_inputs_mask=mention_idx_mask,
            )
            if params["debug"] and params["adversarial_training"]:
                D, _ = cand_encs_index.search(mention_reps.detach().cpu().numpy(), num_neighbors)
                D = torch.tensor(D)
                D = D.flatten()[mask]
                _, scores = reranker(
                    context_input, candidate_input,
                    cand_encs=cand_encs_input, text_encs=mention_reps_input,
                    mention_logits=mention_logits, mention_bounds=mention_bounds,
                    label_input=label_input, gold_mention_idxs=mention_idxs,
                    gold_mention_idx_mask=mention_idx_mask,
                )
                assert ((D - scores[mention_reps.size(0):].to("cpu")) < 0.0005).all()

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
                loss = None  # for GPU mem management
                mention_reps = None
                mention_reps_input = None
                label_input = None
                cand_encs_input = None

                evaluate(
                    reranker, valid_dataloader, params,
                    cand_encs=cand_encs, device=device,
                    logger=logger, faiss_index=cand_encs_flat_index,
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
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, os.path.join(epoch_output_folder_path, "training_state.th"))

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        logger.info("Valid data evaluation")
        results = evaluate(
            reranker, valid_dataloader, params,
            cand_encs=cand_encs, device=device,
            logger=logger, faiss_index=cand_encs_flat_index,
        )
        logger.info("Valid data evaluation -- non-end2end")
        results = evaluate(
            reranker, valid_dataloader, params,
            cand_encs=cand_encs, device=device,
            logger=logger, faiss_index=cand_encs_flat_index,
            joint_mention_detection=False,
        )
        logger.info("Train data evaluation")
        results = evaluate(
            reranker, train_dataloader, params,
            cand_encs=cand_encs, device=device,
            logger=logger, faiss_index=cand_encs_flat_index,
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
        evaluate(params, cand_encs=cand_encs, logger=logger, faiss_index=cand_encs_flat_index)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
