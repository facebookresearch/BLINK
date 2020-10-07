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
from elq.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


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


def get_context_representation_multiple_mentions_left_right(
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


def sort_mentions(
    lst, sort_map=None,
):
    """
    sort_map: {orig_idx: idx in new "sorted" array}
    """
    new_lst = [0 for _ in range(len(lst))]
    for i in range(len(lst)):
        new_lst[sort_map[i]] = lst[i]
    return new_lst


def do_sort(
    sample, orig_idx_to_sort_idx,
):
    sample['mentions'] = sort_mentions(sample['mentions'], orig_idx_to_sort_idx)
    sample['label_id'] = sort_mentions(sample['label_id'], orig_idx_to_sort_idx)
    sample['wikidata_id'] = sort_mentions(sample['wikidata_id'], orig_idx_to_sort_idx)
    sample['entity'] = sort_mentions(sample['entity'], orig_idx_to_sort_idx)
    sample['label'] = sort_mentions(sample['label'], orig_idx_to_sort_idx)


def get_context_representation_multiple_mentions_idxs(
    sample, tokenizer, max_seq_length,
    mention_key, context_key, ent_start_token, ent_end_token,
):
    '''
    Also cuts out mentions beyond that context window

    ASSUMES MENTION_IDXS ARE SORTED!!!!

    Returns:
        List of mention bounds that are [inclusive, exclusive) (make both inclusive later)
        NOTE: 2nd index of mention bound may be outside of max_seq_length-range (must deal with later)
    '''
    mention_idxs = sample["tokenized_mention_idxs"]
    input_ids = sample["tokenized_text_ids"]

    # sort mentions / entities / everything associated
    # [[orig_index, [start, end]], ....] --> sort by start, then end
    sort_tuples = [[i[0], i[1]] for i in sorted(enumerate(mention_idxs), key=lambda x:(x[1][0], x[1][1]))]
    if [tup[1] for tup in sort_tuples] != mention_idxs:
        orig_idx_to_sort_idx = {itm[0]: i for i, itm in enumerate(sort_tuples)}
        assert [tup[1] for tup in sort_tuples] == sort_mentions(mention_idxs, orig_idx_to_sort_idx)
        mention_idxs = [tup[1] for tup in sort_tuples]
        sample['tokenized_mention_idxs'] = mention_idxs
        do_sort(sample, orig_idx_to_sort_idx)
        # TODO SORT EVERYTHING

    # fit leftmost mention, then all of the others that can reasonably fit...
    all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    while all_mention_spans_range[1] - all_mention_spans_range[0] + 2 > max_seq_length:
        if len(mention_idxs) == 1:
            # don't cut further
            assert mention_idxs[0][1] - mention_idxs[0][0] + 2 > max_seq_length
            # truncate mention
            mention_idxs[0][1] = max_seq_length + mention_idxs[0][0] - 2
        else:
            # cut last mention
            mention_idxs = mention_idxs[:len(mention_idxs) - 1]
        all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    
    context_left = input_ids[:all_mention_spans_range[0]]
    all_mention_tokens = input_ids[all_mention_spans_range[0]:all_mention_spans_range[1]]
    context_right = input_ids[all_mention_spans_range[1]:]

    left_quota = (max_seq_length - len(all_mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(all_mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:  # tokens left to add <= quota ON THE LEFT
        if right_add > right_quota:  # add remaining quota to right quota
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:  # tokens left to add <= quota ON THE RIGHT
            left_quota += right_quota - right_add  # add remaining quota to left quota

    if left_quota <= 0:
        left_quota = -len(context_left)  # cut entire list (context_left = [])
    if right_quota <= 0:
        right_quota = 0  # cut entire list (context_right = [])
    input_ids_window = context_left[-left_quota:] + all_mention_tokens + context_right[:right_quota]

    # shift mention_idxs
    if len(input_ids) <= max_seq_length - 2:
        try:
            assert input_ids == input_ids_window
        except:
            import pdb
            pdb.set_trace()
    else:
        assert input_ids != input_ids_window
        cut_from_left = len(context_left) - len(context_left[-left_quota:])
        if cut_from_left > 0:
            # must shift mention_idxs
            for c in range(len(mention_idxs)):
                mention_idxs[c] = [
                    mention_idxs[c][0] - cut_from_left, mention_idxs[c][1] - cut_from_left,
                ]

    input_ids_window = [101] + input_ids_window + [102]
    tokens = tokenizer.convert_ids_to_tokens(input_ids_window)

    # +1 for CLS token
    mention_idxs = [[mention[0]+1, mention[1]+1] for mention in mention_idxs]

    # input_ids = tokenizer.convert_tokens_to_ids(input_ids_window)
    padding = [0] * (max_seq_length - len(input_ids_window))
    input_ids_window += padding
    assert len(input_ids_window) == max_seq_length

    return {
        "tokens": tokens,
        "ids": input_ids_window,
        "mention_idxs": mention_idxs,
        # "pruned_ents": [1 for i in range(len(all_mentions)) if i < len(mention_idxs) else 0],  # pruned last N entities, TODO change if changed
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
    add_mention_bounds=True,
    saved_context_dir=None,
    candidate_token_ids=None,
    params=None,
):
    '''
    Returns /inclusive/ bounds
    '''
    extra_ret_values = {}
    if saved_context_dir is not None and os.path.exists(os.path.join(saved_context_dir, "tensor_tuple.pt")):
        data = torch.load(os.path.join(saved_context_dir, "data.pt"))
        tensor_data_tuple = torch.load(os.path.join(saved_context_dir, "tensor_tuple.pt"))
        return data, tensor_data_tuple, extra_ret_values

    if candidate_token_ids is None and not debug:
        candidate_token_ids = torch.load(params["cand_token_ids_path"])
        if logger: logger.info("Loaded saved entities info")
        extra_ret_values["candidate_token_ids"] = candidate_token_ids

    processed_samples = []

    if debug:
        samples = samples[:200]
    if silent:	
        iter_ = samples
    else:	
        iter_ = tqdm(samples)

    use_world = True

    ent_start_id = tokenizer.convert_tokens_to_ids(ent_start_token)
    ent_end_id = tokenizer.convert_tokens_to_ids(ent_end_token)
    cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    for idx, sample in enumerate(iter_):
        assert not add_mention_bounds, "Adding mention bounds, but we have multiple entities per example"
        if context_key + "_left" in sample:
            context_tokens = get_context_representation_multiple_mentions_left_right(
                sample, tokenizer, max_context_length,
                mention_key, context_key, ent_start_token, ent_end_token,
            )
        else:
            context_tokens = get_context_representation_multiple_mentions_idxs(
                sample, tokenizer, max_context_length,
                mention_key, context_key, ent_start_token, ent_end_token,
            )

        for i in range(len(context_tokens["mention_idxs"])):
            context_tokens["mention_idxs"][i][1] -= 1  # make bounds inclusive

        label = sample[label_key]
        title = sample.get(title_key)
        label_ids = sample.get("label_id")

        if label is None:
            label = [None]
            label_ids = [label_ids]
        # remove those that got pruned off
        if len(label) > len(context_tokens['mention_idxs']):
            label = label[:len(context_tokens['mention_idxs'])]
            label_ids = sample["label_id"][:len(context_tokens['mention_idxs'])]

        if candidate_token_ids is not None:
            token_ids = [[candidate_token_ids[label_id].tolist()] for label_id in label_ids]
            label_tokens = {
                "tokens": "",
                "ids": token_ids,
            }
        elif not params["freeze_cand_enc"]:
            label_tokens = [get_candidate_representation(
                l, tokenizer, max_cand_length, title[i],
            ) for i, l in enumerate(label)]
            label_tokens = {
                k: [label_tokens[l][k] for l in range(len(label_tokens))]
            for k in label_tokens[0]}
        else:
            label_tokens = None

        if isinstance(sample["label_id"], list):
            # multiple candidates
            if len(sample["label_id"]) > len(context_tokens['mention_idxs']):
                sample["label_id"] = sample["label_id"][:len(context_tokens['mention_idxs'])]
            label_idx = [int(id) for id in sample["label_id"]]
        else:
            assert isinstance(sample["label_id"], int) or isinstance(sample["label_id"], str)
            label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
        }
        if not params["freeze_cand_enc"]:
            record["label"] = label_tokens
        record["label_idx"] = label_idx

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            if not params["freeze_cand_encs"]:
                logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
                logger.info(
                    "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
                )
            logger.info("Label_id : %d" % sample["label_idx"])
            if use_world:
                logger.info("Src : %d" % sample["src"][0])

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

        if not params["freeze_cand_enc"]:
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

        if not params["freeze_cand_enc"]:
            cand_vecs, cand_mask = select_field_with_padding(
                processed_samples, "label", "ids", pad_idx=[[0 for _ in range(max_cand_length)]],
            )
            # (bs, max_num_spans, 1, max_cand_length)
            cand_vecs = torch.tensor(cand_vecs, dtype=torch.long)
            cand_mask = torch.tensor(cand_mask, dtype=torch.bool)
            assert (cand_mask == mention_idx_mask).all() or cand_mask.all()
            if logger:
                logger.info("Created candidate IDs vector")
        else:
            cand_vecs = torch.Tensor(context_vecs.size())

        label_idx_vecs, label_idx_mask = select_field_with_padding(processed_samples, "label_idx", pad_idx=-1)
        # (bs, max_num_spans)
        label_idx = torch.tensor(label_idx_vecs, dtype=torch.long)
        label_idx_mask = torch.tensor(label_idx_mask, dtype=torch.bool)
        assert (label_idx_mask == mention_idx_mask).all() or label_idx_mask.all()
        if logger:
            logger.info("Created label IDXs vector")
    # mention_idx_vecs: (bs, max_num_spans, 2), mention_idx_mask: (bs, max_num_spans)
    assert len(mention_idx_vecs.size()) == 3
    # prune mention_idx_vecs to max_context_length
    mention_idx_vecs[mention_idx_vecs >= max_context_length] = (max_context_length - 1)

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
    else:
        tensor_data_tuple = (context_vecs, cand_vecs, label_idx, mention_idx_vecs, mention_idx_mask)
    # save data
    if saved_context_dir is not None and not os.path.exists(os.path.join(saved_context_dir, "tensor_tuple.pt")):
        os.makedirs(saved_context_dir, exist_ok=True)
        torch.save(data, os.path.join(saved_context_dir, "data.pt"))
        torch.save(tensor_data_tuple, os.path.join(saved_context_dir, "tensor_tuple.pt"))
    return data, tensor_data_tuple, extra_ret_values
