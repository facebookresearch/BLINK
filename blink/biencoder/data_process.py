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


def select_field_with_padding(data, key1, key2=None, pad_idx=-1):
    max_len = 0
    selected_list = []
    padding_mask = []
    for example in data:
        if key2 is None:
            selected_list.append(example[key1])
            max_len = max(max_len, len(example[key1]))
        else:
            selected_list.append(example[key1][key2])
            max_len = max(max_len, len(example[key1][key2]))
    for i, entry in enumerate(selected_list):
        # pad to max len
        pad_list = [1 for _ in range(len(entry))] + [0 for _ in range(max_len - len(entry))]
        selected_list[i] += [pad_idx for _ in range(max_len - len(entry))]
        assert len(pad_list) == max_len
        assert len(selected_list[i]) == max_len
        padding_mask.append(pad_list)
    return selected_list, padding_mask


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation_single_mention(
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
        to_subtract = 4 if add_mention_bounds else 2
        if len(mention_tokens) > max_seq_length - to_subtract:	
            # -4 as 2 for ent_start and ent_end, 2 for [CLS] and [SEP]	
            mention_tokens = mention_tokens[:max_seq_length - to_subtract]	
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


def get_context_representation_multiple_mentions(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    all_mentions = sample[mention_key]
    all_context_lefts = sample[context_key + "_left"]
    all_context_rights = sample[context_key + "_right"]

    if len(all_mentions[0]) == 0 and len(all_context_lefts[0]) == 0 and len(all_context_rights[0]) == 0:  # passed in empty string
        context_tokens = ["[CLS]", "[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length
        return {
            "tokens": context_tokens,
            "ids": input_ids,
            "mention_idxs": [],
        }

    mention_tokens = []
    for mention in all_mentions:
        if mention and len(mention) > 0:
            mention_token = tokenizer.tokenize(mention)
            if len(mention_token) > max_seq_length - 2:	
                # -2 for [CLS] and [SEP]
                mention_token = mention_token[:max_seq_length - 2]
            mention_tokens.append(mention_token)
    mention_idxs = []

    assert len(all_context_lefts) == len(all_context_rights)
    assert len(all_context_rights) == len(all_mentions)

    context_tokens = None

    for c in range(len(all_context_lefts)):
        context_left = all_context_lefts[c]
        context_right = all_context_rights[c]

        context_left = tokenizer.tokenize(context_left)
        context_right = tokenizer.tokenize(context_right)

        left_quota = (max_seq_length - len(mention_tokens[c])) // 2 - 1
        right_quota = max_seq_length - len(mention_tokens[c]) - left_quota - 2
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
        context_tokens_itr = (
            context_left[-left_quota:] + mention_tokens[c] + context_right[:right_quota]
        )

        context_tokens_itr = ["[CLS]"] + context_tokens_itr + ["[SEP]"]
        if context_tokens is None:
            context_tokens = context_tokens_itr
        else:
            try:
                assert context_tokens == context_tokens_itr
            except:
                import pdb
                pdb.set_trace()
        mention_idxs.append([
            len(context_left[-left_quota:]) + 1,
            len(context_left[-left_quota:]) + len(mention_tokens[c]) + 1,
        ])

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
        "ids": [input_ids],
    }


def get_context_representation_from_saved(
    sample, max_context_length, ent_start_token, ent_end_token,
    ent_start_id, ent_end_id, cls_token_id, sep_token_id,
    add_mention_bounds, saved_contexts, idx,
):
    # Sanity checks to ensure we have the correct corresponding saved entry
    assert saved_contexts[idx]['mention'] == sample['mention']
    assert saved_contexts[idx]['context_left'] == sample['context_left']
    assert saved_contexts[idx]['context_right'] == sample['context_right']

    # STRIP CLS/SEP tokens and PAD on context_right_ids
    if saved_contexts[idx]['context_left_tokens'][0] == "[CLS]":
        saved_contexts[idx]['context_left_tokens'] = saved_contexts[idx]['context_left_tokens'][1:]
        saved_contexts[idx]['context_left_ids'] = saved_contexts[idx]['context_left_ids'][1:]
    if saved_contexts[idx]['context_right_tokens'][-1] == "[SEP]":
        saved_contexts[idx]['context_right_tokens'] = saved_contexts[idx]['context_right_tokens'][:-1]
        saved_contexts[idx]['context_right_ids'] = saved_contexts[idx]['context_right_ids'][:len(saved_contexts[idx]['context_right_tokens'])]

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

    # MENTION BOUNDARY CUTTING HERE IN ACCORDANCE TO MAX_CONTEXT_LENGTH
    left_quota = (max_context_length - len(saved_contexts[idx]['mention_tokens'])) // 2 - 1
    right_quota = max_context_length - len(saved_contexts[idx]['mention_tokens']) - left_quota - 2
    left_add = len(saved_contexts[idx]['context_left_tokens'])
    right_add = len(saved_contexts[idx]['context_right_tokens'])
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    if left_quota <= 0:	
        context_left = []
        context_left_ids = []
    else:
        context_left = saved_contexts[idx]['context_left_tokens'][-left_quota:]
        context_left_ids = saved_contexts[idx]['context_left_ids'][-left_quota:]
    assert len(context_left) == len(context_left_ids)

    if right_quota <= 0:	
        context_right = []
        context_right_ids = []
    else:
        context_right = saved_contexts[idx]['context_right_tokens'][:right_quota]
        context_right_ids = saved_contexts[idx]['context_right_ids'][:right_quota]
    assert len(context_right) == len(context_right_ids)

    context_tokens = ["[CLS]"] + context_left + saved_contexts[idx]['mention_tokens'] + context_right + ["[SEP]"]
    input_ids = [cls_token_id] + context_left_ids + saved_contexts[idx]['mention_ids'] + context_right_ids + [sep_token_id]

    # add in any additional padding
    padding = [0] * (max_context_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_context_length

    # Get mention / context tokens
    mention_idxs = [
        len(context_left) + 1, len(context_left) + len(saved_contexts[idx]['mention_tokens']),  # make bounds inclusive
    ]
    return {
        "tokens": context_tokens,
        "ids": input_ids,
        "mention_idxs": mention_idxs,
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
    cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    for idx, sample in enumerate(iter_):
        if saved_contexts is not None:
            if len(saved_contexts[idx]['mention_tokens']) + 2 > max_context_length:
                # skip if "[CLS] + mention + [SEP]" is longer
                continue

            all_context_lefts = sample[context_key + "_left"]
            assert isinstance(all_context_lefts, str), "Loading from saved implies this is pretraining data, however have multiple entities per example"

            context_tokens = get_context_representation_from_saved(
                sample=sample,
                max_context_length=max_context_length,
                ent_start_token=ent_start_token,
                ent_end_token=ent_end_token,
                ent_start_id=ent_start_id,
                ent_end_id=ent_end_id,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                add_mention_bounds=add_mention_bounds,
                saved_contexts=saved_contexts,
                idx=idx,
            )

            if do_verify:
                context_tokens_test = get_context_representation_single_mention(
                    sample,
                    tokenizer,
                    max_context_length,
                    mention_key,
                    context_key,
                    ent_start_token,
                    ent_end_token,
                    add_mention_bounds=add_mention_bounds,
                )
                try:
                    context_tokens_test['mention_idxs'][1] -= 1
                    assert context_tokens == context_tokens_test
                except AssertionError:
                    import pdb
                    pdb.set_trace()
        else:
            all_context_lefts = sample[context_key + "_left"]

            if isinstance(all_context_lefts, str):
                sample[context_key + "_left"] = [sample[context_key + "_left"]]
                sample[context_key + "_right"] = [sample[context_key + "_right"]]
                sample[mention_key] = [sample[mention_key]]
            #     context_tokens = get_context_representation_single_mention(
            #         sample,
            #         tokenizer,
            #         max_context_length,
            #         mention_key,
            #         context_key,
            #         ent_start_token,
            #         ent_end_token,
            #         add_mention_bounds=add_mention_bounds,
            #     )
            # elif isinstance(all_context_lefts, list):
            assert not add_mention_bounds, "Adding mention bounds, but we have multiple entities per example"
            context_tokens = get_context_representation_multiple_mentions(
                sample, tokenizer, max_context_length,
                mention_key, context_key, ent_start_token, ent_end_token,
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
                    'context_right_ids': context_tokens['ids'][context_tokens['mention_idxs'][1]:],  # has padding
                    'context_left': sample['context_left'],
                    'mention': sample['mention'],
                    'context_right': sample['context_right'],
                }
                all_saved_encodings.append(saved_encodings)
            for i in range(len(context_tokens["mention_idxs"])):
                context_tokens["mention_idxs"][i][1] -= 1  # make bounds inclusive

        label = sample[label_key]
        title = sample.get(title_key, None)
        if get_cached_representation:
            assert candidate_token_ids is not None
            assert entity2id is not None
            # TODO REVERT UNLESS IS LIST
            if isinstance(sample["label_id"], list):
                token_ids = [candidate_token_ids[entity2id[
                    entity
                ]].tolist() for entity in sample.get('entity', None)]
                import pdb
                pdb.set_trace()
            else:
                import pdb
                pdb.set_trace()
                token_ids = candidate_token_ids[entity2id[
                    sample.get('entity', None)
                ]].tolist()
            label_tokens = {
                "tokens": "",
                "ids": token_ids,
            }
        else:
            if label is None:
                label = [None]
                sample["label_id"] = [sample["label_id"]]
            label_tokens = [get_candidate_representation(
                l, tokenizer, max_cand_length, title,
            ) for l in label]
            label_tokens = {
                k: [label_tokens[l][k] for l in range(len(label_tokens))]
            for k in label_tokens[0]}
        if isinstance(sample["label_id"], list):
            label_idx = [int(id) for id in sample["label_id"]]
        else:
            assert isinstance(sample["label_id"], int) or isinstance(sample["label_id"], str)
            label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": label_idx,
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
            logger.info("Label_id : %d" % sample["label_idx"])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    if logger:
        logger.info("Created context IDs vector")
    if isinstance(processed_samples[0]["context"]["mention_idxs"][0], int):
        mention_idx_vecs = torch.tensor(
            select_field(processed_samples, "context", "mention_idxs"), dtype=torch.long,
        ).unsqueeze(1)
        mention_idx_mask = torch.ones(mention_idx_vecs.size(0), dtype=torch.bool).unsqueeze(-1)
        if logger:
            logger.info("Created mention positions vector")

        cand_vecs = torch.tensor(
            select_field(processed_samples, "label", "ids"), dtype=torch.long,
        )
        if logger:
            logger.info("Created candidate IDs vector")

        label_idx = torch.tensor(
            select_field(processed_samples, "label_idx"), dtype=torch.long,
        ).unsqueeze(-1)
        if logger:
            logger.info("Created label IDXs vector")
    else:
        mention_idx_vecs, mention_idx_mask = select_field_with_padding(
            processed_samples, "context", "mention_idxs", pad_idx=[0,1],  #ensure is a well-formed span
        )
        # (bs, max_num_spans, 2)
        mention_idx_vecs = torch.tensor(mention_idx_vecs, dtype=torch.long)
        # (bs, max_num_spans)
        mention_idx_mask = torch.tensor(mention_idx_mask, dtype=torch.bool)

        cand_vecs, cand_mask = select_field_with_padding(
            processed_samples, "label", "ids", pad_idx=[[0 for _ in range(max_cand_length)]],
        )
        # (bs, max_num_spans, 1, max_cand_length)
        cand_vecs = torch.tensor(cand_vecs, dtype=torch.long)
        cand_mask = torch.tensor(cand_mask, dtype=torch.bool)
        assert (cand_mask == mention_idx_mask).all() or cand_mask.all()
        if logger:
            logger.info("Created candidate IDs vector")

        label_idx_vecs, label_idx_mask = select_field_with_padding(processed_samples, "label_idx", pad_idx=-1)
        # (bs, max_num_spans)
        label_idx = torch.tensor(label_idx_vecs, dtype=torch.long)
        label_idx_mask = torch.tensor(label_idx_mask, dtype=torch.bool)
        assert (label_idx_mask == mention_idx_mask).all() or label_idx_mask.all()
        if logger:
            logger.info("Created label IDXs vector")
    # mention_idx_vecs: (bs, max_num_spans, 2), mention_idx_mask: (bs, max_num_spans)
    assert len(mention_idx_vecs.size()) == 3

    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
        if logger:
            logger.info("Created source vector")
    
    data = {
        "context_vecs": context_vecs,
        "mention_idx_vecs": mention_idx_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data_tuple = (context_vecs, cand_vecs, src_vecs, label_idx, mention_idx_vecs, mention_idx_mask)
        # tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx, mention_idx_vecs)
    else:
        tensor_data_tuple = (context_vecs, cand_vecs, label_idx, mention_idx_vecs, mention_idx_mask)
        # tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx, mention_idx_vecs)
    if logger:
        logger.info("Created tensor dataset")
    return data, tensor_data_tuple
