# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    add_mention_bounds=True,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        if len(mention_tokens) > max_seq_length - 4:	
            # -4 as 2 for ent_start and ent_end, 2 for [CLS] and [SEP]	
            mention_tokens = mention_tokens[:max_seq_length - 4]	
        if add_mention_bounds:
            mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add
            
    if left_quota <= 0:	
        context_left = []	
    if right_quota <= 0:	
        context_right = []
    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    mention_idxs = [
        len(context_left[-left_quota:]) + 1,
        len(context_left[-left_quota:]) + len(mention_tokens) + 1,
    ]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
        "mention_idxs": mention_idxs,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
    add_mention_bounds=True,  # TODO change
    get_cached_representation=True,
    candidate_token_ids=None,
    entity2id=None,
    saved_context_file=None,  # file with set of contexts
    start_idx=0,
    end_idx=-1,
    do_verify=False,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    print("Start_idx: {}, End_idx: {}".format(start_idx, end_idx))
    if end_idx == -1:
        end_idx = len(samples)
    if silent:
        iter_ = samples[start_idx:end_idx]
    else:
        iter_ = tqdm(samples[start_idx:end_idx])

    use_world = True
    saved_contexts = None
    if get_cached_representation and saved_context_file is not None and os.path.exists(saved_context_file):
        saved_contexts = json.load(open(saved_context_file))
    all_saved_encodings = []

    ent_start_id = tokenizer.convert_tokens_to_ids(ent_start_token)
    ent_end_id = tokenizer.convert_tokens_to_ids(ent_end_token)
    for idx, sample in enumerate(iter_):
        if saved_contexts is not None:
            assert saved_contexts[idx]['mention'] == sample['mention']
            assert saved_contexts[idx]['context_left'] == sample['context_left']
            assert saved_contexts[idx]['context_right'] == sample['context_right']
            if add_mention_bounds:
                saved_contexts[idx]['mention_tokens'] = (
                    [ent_start_token] +
                    saved_contexts[idx]['mention_tokens'] +
                    [ent_end_token]
                )
                saved_contexts[idx]['mention_ids'] = (
                    [ent_start_id] +
                    saved_contexts[idx]['mention_ids'] +
                    [ent_end_id]
                )
            mention_idxs = [
                len(saved_contexts[idx]['context_left_ids']),
                len(saved_contexts[idx]['context_left_ids']) + len(saved_contexts[idx]['mention_tokens']) - 1,  # make bounds inclusive
            ]
            # TODO VERIFY THE SAVED CONTEXTS
            context_tokens = {
                "tokens": saved_contexts[idx]['context_left_tokens'] + saved_contexts[idx]['mention_tokens'] + saved_contexts[idx]['context_right_tokens'],
                "ids": saved_contexts[idx]['context_left_ids'] + saved_contexts[idx]['mention_ids'] + saved_contexts[idx]['context_right_ids'],
                "mention_idxs": mention_idxs,
            }
            if do_verify:
                context_tokens_test = get_context_representation(
                    sample,
                    tokenizer,
                    max_context_length,
                    mention_key,
                    context_key,
                    ent_start_token,
                    ent_end_token,
                    add_mention_bounds=add_mention_bounds,
                )
                assert context_tokens == context_tokens_test
        else:
            context_tokens = get_context_representation(
                sample,
                tokenizer,
                max_context_length,
                mention_key,
                context_key,
                ent_start_token,
                ent_end_token,
                add_mention_bounds=add_mention_bounds,
            )
            # save cached representation
            if get_cached_representation:
                mention_ids = context_tokens['ids'][context_tokens['mention_idxs'][0]:context_tokens['mention_idxs'][1]]
                mention_tokens = context_tokens['tokens'][context_tokens['mention_idxs'][0]:context_tokens['mention_idxs'][1]]
                if add_mention_bounds:
                    mention_ids = mention_ids[1:-1]
                    mention_tokens = mention_tokens[1:-1]
                saved_encodings = {
                    'context_left_tokens': context_tokens['tokens'][:context_tokens['mention_idxs'][0]],
                    'mention_tokens': mention_tokens,
                    'context_right_tokens': context_tokens['tokens'][context_tokens['mention_idxs'][1]:],
                    'context_left_ids': context_tokens['ids'][:context_tokens['mention_idxs'][0]],
                    'mention_ids': mention_ids,
                    'context_right_ids': context_tokens['ids'][context_tokens['mention_idxs'][1]:],
                    'context_left': sample['context_left'],
                    'mention': sample['mention'],
                    'context_right': sample['context_right'],
                }
                all_saved_encodings.append(saved_encodings)
            context_tokens["mention_idxs"][1] -= 1  # make boundsinclusive

        label = sample[label_key]
        title = sample.get(title_key, None)
        if get_cached_representation:
            assert candidate_token_ids is not None
            assert entity2id is not None
            token_ids = candidate_token_ids[entity2id[
                sample.get('entity', None)
            ]].tolist()
            label_tokens = {
                "tokens": "",
                "ids": token_ids,
            }
        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title,
            )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if logger:
        logger.info("Finished loading data")
    # save memory
    saved_contexts = None

    # saved cached file
    if get_cached_representation and not os.path.exists(saved_context_file):
        json.dump(all_saved_encodings, open("{}_{}_{}.json".format(saved_context_file, start_idx, end_idx), "w"))
    # save memory
    all_saved_encodings = []

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            if use_world:
                logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    if logger:
        logger.info("Created context IDs vector")
    mention_idx_vecs = torch.tensor(
        select_field(processed_samples, "context", "mention_idxs"), dtype=torch.long,
    )
    if logger:
        logger.info("Created mention positions vector")
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if logger:
        logger.info("Created candidate IDs vector")
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
        if logger:
            logger.info("Created source vector")
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    if logger:
        logger.info("Created label IDXs vector")
    data = {
        "context_vecs": context_vecs,
        "mention_idx_vecs": mention_idx_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx, mention_idx_vecs)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx, mention_idx_vecs)
    if logger:
        logger.info("Created tensor dataset")
    return data, tensor_data
