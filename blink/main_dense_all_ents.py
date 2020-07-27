# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys
import faiss

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored
import torch.nn.functional as F

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder, to_bert_input
from blink.biencoder.data_process import (
    process_mention_data,
    get_context_representation_single_mention,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
import math

import blink.vcg_utils
from blink.vcg_utils.mention_extraction import extract_entities
from blink.vcg_utils.measures import entity_linking_tp_with_overlap

import os
import sys
from tqdm import tqdm
import pdb
import time


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) : ]
    else:
        msg = input_sentence
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(idx, sample, e_id, e_title, e_text, e_url, show_url=False):	
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))	
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])	
    if show_url:	
        to_print += "url:{}\n".format(e_url)	
    print(to_print)


def _load_candidates(
    entity_catalogue, entity_encoding, entity_token_ids, biencoder, max_seq_length,
    get_kbids=True, logger=None,
):
    candidate_encoding = torch.load(entity_encoding)
    candidate_token_ids = torch.load(entity_token_ids)
    return candidate_encoding, candidate_token_ids


def _get_test_samples(
    test_filename, test_entities_path, logger,
    qa_data=False, do_ner="none", debug=False,
):
    """
    Parses jsonl format with one example per line
    Each line of the following form

    IF HAVE LABELS
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
        "mentions": [[19, 23], [7, 15]],
        "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
        "tokenized_mention_idxs": [[4, 5], [2, 3]],
        "label_id": [10902, 28422],
        "wikidata_id": ["Q1397", "Q132050"],
        "entity": ["Ohio", "Governor"],
        "label": [list of wikipedia descriptions]
    }

    IF NO LABELS (JUST PREDICTION)
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    test_samples = []
    unknown_entity_samples = []
    num_unknown_entity_samples = 0
    num_no_gold_entity = 0
    ner_errors = 0

    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        sample_idx = 0
        do_setup_samples = True
        for i, line in enumerate(tqdm(lines)):
            record = json.loads(line)
            test_samples.append(record)

    return test_samples, num_unknown_entity_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params, logger):
    """
    Samples: list of examples, each of the form--

    IF HAVE LABELS
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
        "mentions": [[19, 23], [7, 15]],
        "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
        "tokenized_mention_idxs": [[4, 5], [2, 3]],
        "label_id": [10902, 28422],
        "wikidata_id": ["Q1397", "Q132050"],
        "entity": ["Ohio", "Governor"],
        "label": [list of wikipedia descriptions]
    }

    IF NO LABELS (JUST PREDICTION)
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    if 'label_id' in samples[0]:
        # have labels
        tokens_data, tensor_data_tuple, _ = process_mention_data(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=biencoder_params["max_context_length"],
            max_cand_length=biencoder_params["max_cand_length"],
            silent=False,
            logger=logger,
            debug=biencoder_params["debug"],
            add_mention_bounds=(not biencoder_params.get("no_mention_bounds", False)),
        )
    else:
        samples_text_tuple = []
        max_seq_len = 0
        for sample in samples:
            samples_text_tuple
            encoded_sample = [101] + tokenizer.encode(sample['text']) + [102]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            samples_text_tuple.append(encoded_sample + [0 for _ in range(biencoder_params["max_context_length"] - len(encoded_sample))])
        tensor_data_tuple = [torch.tensor(samples_text_tuple)]
    tensor_data = TensorDataset(*tensor_data_tuple)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _batch_reshape_mask_left(
    input_t, selected, pad_idx=0, left_align_mask=None
):
    """
    Left-aligns all ``selected" values in input_t, which is a batch of examples.
        - input_t: >=2D tensor (N, M, *)
        - selected: 2D torch.Bool tensor, 2 dims same size as first 2 dims of `input_t` (N, M) 
        - pad_idx represents the padding to be used in the output
        - left_align_mask: if already precomputed, pass the alignment mask in
    Example:
        input_t  = [[1,2,3,4],[5,6,7,8]]
        selected = [[0,1,0,1],[1,1,0,1]]
        output   = [[2,4,0],[5,6,8]]
    """
    batch_num_selected = selected.sum(1)
    max_num_selected = batch_num_selected.max()

    # (bsz, 2)
    repeat_freqs = torch.stack([batch_num_selected, max_num_selected - batch_num_selected], dim=-1)
    # (bsz x 2,)
    repeat_freqs = repeat_freqs.view(-1)

    if left_align_mask is None:
        # (bsz, 2)
        left_align_mask = torch.zeros(input_t.size(0), 2).to(input_t.device).bool()
        left_align_mask[:,0] = 1
        # (bsz x 2,): [1,0,1,0,...]
        left_align_mask = left_align_mask.view(-1)
        # (bsz x max_num_selected,): [1 xrepeat_freqs[0],0 x(M-repeat_freqs[0]),1 xrepeat_freqs[1],0 x(M-repeat_freqs[1]),...]
        left_align_mask = left_align_mask.repeat_interleave(repeat_freqs)
        # (bsz, max_num_selected)
        left_align_mask = left_align_mask.view(-1, max_num_selected)

    # reshape to (bsz, max_num_selected, *)
    input_reshape = torch.Tensor(left_align_mask.size() + input_t.size()[2:]).to(input_t.device, input_t.dtype).fill_(pad_idx)
    input_reshape[left_align_mask] = input_t[selected]
    # (bsz, max_num_selected, *); (bsz, max_num_selected)
    return input_reshape, left_align_mask


def _run_biencoder(
    args, biencoder, dataloader, candidate_encoding, samples,
    top_k=100, device="cpu", jointly_extract_mentions=False,
    sample_to_all_context_inputs=None, num_mentions=10,  # TODO don't hardcode
    mention_classifier_threshold=0.0,
    cand_encs_flat_index=None,
):
    """
    Returns: tuple
        labels (List[int]) [(max_num_mentions_gold) x exs]: gold labels -- returns None if no labels
        nns (List[Array[int]]) [(# of pred mentions, cands_per_mention) x exs]: predicted entity IDs in each example
        dists (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: scores of each entity in nns
        pred_mention_bounds (List[Array[int]]) [(# of pred mentions, 2) x exs]: predicted mention boundaries in each examples
        mention_scores (List[Array[float]]) [(# of pred mentions,) x exs]: mention score logit
        cand_scores (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: candidate score logit
    """
    biencoder.model.eval()
    biencoder_model = biencoder.model
    if hasattr(biencoder.model, "module"):
        biencoder_model = biencoder.model.module

    context_inputs = []
    nns = []
    dists = []
    mention_dists = []
    pred_mention_bounds = []
    mention_scores = []
    cand_scores = []
    sample_idx = 0
    ctxt_idx = 0
    new_samples = samples
    new_sample_to_all_context_inputs = sample_to_all_context_inputs
    if jointly_extract_mentions:
        new_samples = []
        new_sample_to_all_context_inputs = []
    label_ids = None
    for step, batch in enumerate(tqdm(dataloader)):
        context_input = batch[0]
        with torch.no_grad():
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(context_input, biencoder.NULL_IDX)
            if device != "cpu":
                token_idx_ctxt = token_idx_ctxt.to(device)
                segment_idx_ctxt = segment_idx_ctxt.to(device)
                mask_ctxt = mask_ctxt.to(device)
            
            '''
            PREPARE INPUTS
            '''
            # (bsz, seqlen, embed_dim)
            context_encoding, _, _ = biencoder_model.context_encoder.bert_model(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

            '''
            GET MENTION SCORES
            '''
            # (num_total_mentions,); (num_total_mentions,)
            mention_logits, mention_bounds = biencoder_model.classification_heads['mention_scores'](context_encoding, mask_ctxt)

            '''
            PRUNE MENTIONS BASED ON SCORES (for each instance in batch, TOP_K (>= -inf) OR THRESHOLD)
            '''
            '''
            topK_mention_scores, mention_pos = torch.cat([torch.arange(), mention_logits.topk(top_k, dim=1)])
            mention_pos = mention_pos.flatten()
            '''
            # DIM (num_total_mentions, embed_dim)
            # (bsz, top_k); (bsz, top_k)
            top_mention_logits, mention_pos = mention_logits.topk(top_k, sorted=True)
            # 2nd part of OR for if nothing is > 0
            # DIM (bsz, top_k, 2)
            #   [:,:,0]: index of batch
            #   [:,:,1]: index into top mention in mention_bounds
            mention_pos = torch.stack([torch.arange(mention_pos.size(0)).to(mention_pos.device).unsqueeze(-1).expand_as(mention_pos), mention_pos], dim=-1)
            # DIM (bsz, top_k)
            top_mention_pos_mask = torch.sigmoid(top_mention_logits) > mention_classifier_threshold
            # DIM (total_possible_mentions, 2)
            #   tuples of [index of batch, index into mention_bounds] of what mentions to include
            mention_pos = mention_pos[top_mention_pos_mask | (
                # If nothing is > threshold, use topK that are > -inf (2nd part of OR)
                ((top_mention_pos_mask.sum(1) == 0).unsqueeze(-1)) & (top_mention_logits > -float("inf"))
            )]  #(mention_pos_2_mask.sum(1) == 0).unsqueeze(-1)]
            mention_pos = mention_pos.view(-1, 2)  #2297 [45,11]
            # DIM (bs, total_possible_mentions)
            #   mask of possible logits
            mention_pos_mask = torch.zeros(mention_logits.size(), dtype=torch.bool).to(mention_pos.device)
            mention_pos_mask[mention_pos[:,0], mention_pos[:,1]] = 1
            # DIM (bs, max_num_pred_mentions, 2)
            chosen_mention_bounds, left_align_mask = _batch_reshape_mask_left(mention_bounds, mention_pos_mask, pad_idx=0)
            # DIM (bs, max_num_pred_mentions)
            chosen_mention_logits, _ = _batch_reshape_mask_left(mention_logits, mention_pos_mask, pad_idx=-float("inf"), left_align_mask=left_align_mask)

            '''
            GET CANDIDATE SCORES + TOP CANDIDATES PER MENTION
            '''
            # (bs, max_num_pred_mentions, embed_dim)
            embedding_ctxt = biencoder_model.classification_heads['get_context_embeds'](context_encoding, chosen_mention_bounds)
            if biencoder_model.linear_compression is not None:
                embedding_ctxt = biencoder_model.linear_compression(embedding_ctxt)
            # (all_pred_mentions_batch, embed_dim)
            embedding_ctxt = embedding_ctxt[left_align_mask]
            # DIM (num_total_mentions, num_candidates)
            # TODO search for topK entities with FAISS
            try:
                # start_time = time.time()
                if embedding_ctxt.size(0) > 1:
                    embedding_ctxt = embedding_ctxt.squeeze(0)
                # DIM (all_pred_mentions_batch, all_cand_entities)
                cand_logits = embedding_ctxt.mm(candidate_encoding.to(device).t())
                # end_time = time.time()
                # DIM (all_pred_mentions_batch, 10); (all_pred_mentions_batch, 10)
                top_cand_logits_shape, top_cand_indices_shape = cand_logits.topk(10, dim=-1, sorted=True)
            except:
                # for memory savings, go through one chunk of candidates at a time
                SPLIT_SIZE=1000000
                done=False
                while not done:
                    top_cand_logits_list = []
                    top_cand_indices_list = []
                    max_chunk = int(len(candidate_encoding) / SPLIT_SIZE)
                    for chunk_idx in range(max_chunk):
                        try:
                            # DIM (num_total_mentions, 10); (num_total_mention, 10)
                            top_cand_logits, top_cand_indices = embedding_ctxt.mm(candidate_encoding[chunk_idx*SPLIT_SIZE:(chunk_idx+1)*SPLIT_SIZE].to(device).t().contiguous()).topk(10, dim=-1, sorted=True)
                            top_cand_logits_list.append(top_cand_logits)
                            top_cand_indices_list.append(top_cand_indices + chunk_idx*SPLIT_SIZE)
                            if len((top_cand_indices_list[chunk_idx] < 0).nonzero()) > 0:
                                import pdb
                                pdb.set_trace()
                        except:
                            SPLIT_SIZE = int(SPLIT_SIZE/2)
                            break
                    if len(top_cand_indices_list) == max_chunk:
                        # DIM (num_total_mentions, 10); (num_total_mentions, 10) --> top_top_cand_indices_shape indexes into top_cand_indices
                        top_cand_logits_shape, top_top_cand_indices_shape = torch.cat(top_cand_logits_list, dim=-1).topk(10, dim=-1, sorted=True)
                        # make indices index into candidate_encoding
                        # DIM (num_total_mentions, max_chunk*10)
                        all_top_cand_indices = torch.cat(top_cand_indices_list, dim=-1)
                        # DIM (num_total_mentions, 10)
                        top_cand_indices_shape = all_top_cand_indices.gather(-1, top_top_cand_indices_shape)
                        done = True

            # DIM (bs, max_num_pred_mentions, 10)
            top_cand_logits = torch.zeros(chosen_mention_logits.size(0), chosen_mention_logits.size(1), top_cand_logits_shape.size(-1)).to(
                top_cand_logits_shape.device, top_cand_logits_shape.dtype)
            top_cand_logits[left_align_mask] = top_cand_logits_shape
            top_cand_indices = torch.zeros(chosen_mention_logits.size(0), chosen_mention_logits.size(1), top_cand_indices_shape.size(-1)).to(
                top_cand_indices_shape.device, top_cand_indices_shape.dtype)
            top_cand_indices[left_align_mask] = top_cand_indices_shape

            '''
            COMPUTE FINAL SCORES FOR EACH CAND-MENTION PAIR + PRUNE USING IT
            '''
            # Has NAN for impossible mentions...
            # DIM (bs, max_num_pred_mentions, 10)
            scores = torch.log_softmax(top_cand_logits, -1)
            if args.final_thresholding != "top_entity_by_mention":
                # DIM (num_total_mentions, num_candidates)
                # log p(entity && mb) = log [p(entity|mention bounds) * p(mention bounds)] = log p(e|mb) + log p(mb)
                # scores += torch.sigmoid(mention_logits)[mention_pos_mask].unsqueeze(-1)
                # DIM (num_total_mentions, num_candidates)
                # scores = torch.log_softmax(cand_scores, 1) + torch.sigmoid(mention_scores)
                # Is NAN for impossible mentions...
                # DIM (bs, max_num_pred_mentions, 10)
                scores += torch.sigmoid(chosen_mention_logits.unsqueeze(-1)).log()
            # mention_scores = mention_scores.expand_as(cand_scores)

            '''
            DON'T NEED TO RESORT BY NEW SCORE -- DISTANCE PRESERVING (largest entity score still be largest entity score)
            '''
    
            for idx in range(len(batch[0])):
                # TODO do with masking....!!!
                # [(seqlen) x exs] <= (bsz, seqlen)
                context_inputs.append(context_input[idx][mask_ctxt[idx]].data.cpu().numpy())
                if len(top_cand_indices[idx][top_cand_indices[idx] < 0]) > 0:
                    import pdb
                    pdb.set_trace()
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
                nns.append(top_cand_indices[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
                dists.append(scores[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, 2) x exs] <= (bsz, max_num_mentions=top_k, 2)
                pred_mention_bounds.append(chosen_mention_bounds[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions,) x exs] <= (bsz, max_num_mentions=top_k)
                mention_scores.append(chosen_mention_logits[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
                cand_scores.append(top_cand_logits[idx][left_align_mask[idx]].data.cpu().numpy())

    return nns, dists, pred_mention_bounds, mention_scores, cand_scores  #, new_samples, new_sample_to_all_context_inputs


def get_predictions(
    args, dataloader, biencoder_params, samples, nns,
    dists, mention_scores, cand_scores,
    pred_mention_bounds, id2title, threshold=-2.9,
):
    """
    Arguments:
        args, dataloader, biencoder_params, samples, nns, dists, pred_mention_bounds
    Returns:
        all_entity_preds,
        num_correct, num_predicted, num_gold,
        num_correct_from_input_window, num_gold_from_input_window
    """

    # save biencoder predictions and print precision/recalls
    num_correct = 0
    num_predicted = 0
    num_gold = 0
    num_correct_from_input_window = 0
    num_gold_from_input_window = 0
    all_entity_preds = []

    f = errors_f = None
    if getattr(args, 'save_preds_dir', None) is not None:
        save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
        f = open(save_biencoder_file, 'w')
        errors_f = open(os.path.join(args.save_preds_dir, 'biencoder_errors.jsonl'), 'w')

    # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
    # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
    # cand_scores (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # mention_scores (List[Array[float]]) [(num_pred_mentions,) x exs])
    for batch_num, batch_data in enumerate(dataloader):
        batch_context = batch_data[0]
        if len(batch_data) > 1:
            _, batch_cands, batch_label_ids, batch_mention_idxs, batch_mention_idx_masks = batch_data
        for b in range(len(batch_context)):
            i = batch_num * biencoder_params['eval_batch_size'] + b
            sample = samples[i]
            input_context = batch_context[b][batch_context[b] != 0].tolist()  # filter out padding

            # (num_pred_mentions, cands_per_mention)
            cands_mask = (dists[i][:,0] != -1) & (dists[i][:,0] == dists[i][:,0])
            pred_entity_list = nns[i][cands_mask]
            if len(pred_entity_list) > 0:
                e_id = pred_entity_list[0]
            distances = dists[i][cands_mask]
            # (num_pred_mentions, 2)
            entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]
            utterance = sample['text']

            '''
            get top for each mention bound, w/out duplicates
            # TOP-1
            all_pred_entities = pred_entity_list[:,:1]
            e_mention_bounds = entity_mention_bounds_idx[:1].tolist()
            # '''
            if args.final_thresholding == "joint_0":
                # THRESHOLDING
                assert utterance is not None
                top_mentions_mask = (distances[:,0] > threshold)
                _, sort_idxs = torch.tensor(distances[:,0][top_mentions_mask]).sort(descending=True)
                # cands already sorted by score
                all_pred_entities = pred_entity_list[:,0][top_mentions_mask]
                e_mention_bounds = entity_mention_bounds_idx[top_mentions_mask]
                chosen_distances = distances[:,0][top_mentions_mask]
                if len(all_pred_entities) >= 2:
                    all_pred_entities = all_pred_entities[sort_idxs]
                    e_mention_bounds = e_mention_bounds[sort_idxs]
                    chosen_distances = chosen_distances[sort_idxs]
            elif args.final_thresholding == "top_joint_by_mention" or args.final_thresholding == "top_entity_by_mention":
                if len(entity_mention_bounds_idx[i]) == 0:
                    e_mention_bounds_idxs = []
                else:
                    # 1 PER BOUND
                    try:
                        e_mention_bounds_idxs = [np.where(entity_mention_bounds_idx == j)[0][0] for j in range(len(sample['context_left']))]
                    except:
                        import pdb
                        pdb.set_trace()
                # sort bounds
                e_mention_bounds_idxs.sort()
                all_pred_entities = []
                e_mention_bounds = []
                for bound_idx in e_mention_bounds_idxs:
                    if pred_entity_list[bound_idx] not in all_pred_entities:
                        all_pred_entities.append(pred_entity_list[bound_idx])
                        e_mention_bounds.append(entity_mention_bounds_idx[bound_idx])

            # prune mention overlaps
            e_mention_bounds_pruned = []
            all_pred_entities_pruned = []
            mention_masked_utterance = np.zeros(len(input_context))
            # ensure well-formed-ness, prune overlaps
            # greedily pick highest scoring, then prune all overlapping
            for idx, mb in enumerate(e_mention_bounds):
                mb[1] += 1  # prediction was inclusive, now make exclusive
                # check if in existing mentions
                try:
                    if mention_masked_utterance[mb[0]:mb[1]].sum() >= 1:
                        continue
                except:
                    import pdb
                    pdb.set_trace()
                e_mention_bounds_pruned.append(mb)
                all_pred_entities_pruned.append(all_pred_entities[idx])
                mention_masked_utterance[mb[0]:mb[1]] = 1

            input_context = input_context[1:-1]  # remove BOS and sep
            pred_triples = [(
                # sample['all_gold_entities'][i],
                str(all_pred_entities_pruned[j]),
                int(e_mention_bounds_pruned[j][0]) - 1,  # -1 for BOS
                int(e_mention_bounds_pruned[j][1]) - 1,
            ) for j in range(len(all_pred_entities_pruned))]

            entity_results = {
                "id": sample["id"],
                "text": sample["text"],
                "scores": chosen_distances.tolist(),
            }

            if 'label_id' in sample:
                # Get LABELS
                input_mention_idxs = batch_mention_idxs[b][batch_mention_idx_masks[b]].tolist()
                input_label_ids = batch_label_ids[b][batch_label_ids[b] != -1].tolist()
                assert len(input_label_ids) == len(input_mention_idxs)
                gold_mention_bounds = [
                    sample['text'][ment[0]-10:ment[0]] + "[" + sample['text'][ment[0]:ment[1]] + "]" + sample['text'][ment[1]:ment[1]+10]
                    for ment in sample['mentions']
                ]

                # GET ALIGNED MENTION_IDXS (input is slightly different to model) between ours and gold labels -- also have to account for BOS
                gold_input = sample['tokenized_text_ids']
                # return first instance of my_input in gold_input
                for my_input_start in range(len(gold_input)):
                    if (
                        gold_input[my_input_start] == input_context[0] and
                        gold_input[my_input_start:my_input_start+len(input_context)] == input_context
                    ):
                        break

                # add alignment factor (my_input_start) to predicted mention triples
                pred_triples = [(
                    triple[0], triple[1] + my_input_start, triple[2] + my_input_start,
                ) for triple in pred_triples]
                gold_triples = [(
                    str(sample['label_id'][j]),
                    sample['tokenized_mention_idxs'][j][0], sample['tokenized_mention_idxs'][j][1],
                ) for j in range(len(sample['label_id']))]
                num_overlap = entity_linking_tp_with_overlap(gold_triples, pred_triples)
                num_correct += num_overlap
                num_predicted += len(all_pred_entities_pruned)
                num_gold += len(sample["label_id"])

                # compute number correct given the input window
                pred_input_window_triples = [(
                    # sample['all_gold_entities'][i],
                    str(all_pred_entities_pruned[j]),
                    int(e_mention_bounds_pruned[j][0]), int(e_mention_bounds_pruned[j][1]),
                ) for j in range(len(all_pred_entities_pruned))]
                gold_input_window_triples = [(
                    str(input_label_ids[j]),
                    input_mention_idxs[j][0], input_mention_idxs[j][1] + 1,
                ) for j in range(len(input_label_ids))]
                num_correct_from_input_window += entity_linking_tp_with_overlap(gold_input_window_triples, pred_input_window_triples)
                num_gold_from_input_window += len(input_mention_idxs)

                for triple in pred_triples:
                    if triple[0] not in id2title:
                        import pdb
                        pdb.set_trace()
                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "gold_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in gold_triples
                    ],
                    "pred_triples": pred_triples,
                    "gold_triples": gold_triples,
                    "tokens": input_context,
                })

                if errors_f is not None and (num_overlap != len(gold_triples) or num_overlap != len(pred_triples)):
                    errors_f.write(json.dumps(entity_results) + "\n")
            else:
                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(input_context[triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "pred_triples": pred_triples,
                    "tokens": input_context,
                })

            all_entity_preds.append(entity_results)
            if f is not None:
                f.write(
                    json.dumps(entity_results) + "\n"
                )
    
    if f is not None:
        f.close()
        errors_f.close()
    return all_entity_preds, num_correct, num_predicted, num_gold, num_correct_from_input_window, num_gold_from_input_window


def _retrieve_from_saved_biencoder_outs(save_preds_dir):
    nns = np.load(os.path.join(args.save_preds_dir, "biencoder_nns.npy"), allow_pickle=True)
    # dists = np.load(os.path.join(args.save_preds_dir, "biencoder_dists.npy"), allow_pickle=True)
    pred_mention_bounds = np.load(os.path.join(args.save_preds_dir, "biencoder_mention_bounds.npy"), allow_pickle=True)
    cand_scores = np.load(os.path.join(args.save_preds_dir, "biencoder_cand_scores.npy"), allow_pickle=True)
    mention_scores = np.load(os.path.join(args.save_preds_dir, "biencoder_mention_scores.npy"), allow_pickle=True)

    # TODO delete
    dists = 
    return nns, dists, pred_mention_bounds, cand_scores, mention_scores


def load_models(args, logger):
    # load biencoder model
    logger.info("loading biencoder model")
    try:
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(args.biencoder_config) as json_file:
            for line in json_file:
                line = line.replace("'", "\"")
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                biencoder_params = json.loads(line)
                break
    biencoder_params["path_to_model"] = args.biencoder_model
    biencoder_params["eval_batch_size"] = args.eval_batch_size
    biencoder_params["no_cuda"] = not args.use_cuda
    if biencoder_params["no_cuda"]:
        biencoder_params["data_parallel"] = False
    biencoder_params["load_cand_enc_only"] = False
    if getattr(args, 'max_context_length', None) is not None:
        biencoder_params["max_context_length"] = args.max_context_length
    # biencoder_params["mention_aggregation_type"] = args.mention_aggregation_type
    biencoder = load_biencoder(biencoder_params)
    if not args.use_cuda and type(biencoder.model).__name__ == 'DataParallel':
        biencoder.model = biencoder.model.module
    elif args.use_cuda and type(biencoder.model).__name__ != 'DataParallel':
        biencoder.model = torch.nn.DataParallel(biencoder.model)

    # load candidate entities
    logger.info("loading candidate entities")

    if args.debug_biencoder:
        candidate_encoding, candidate_token_ids = _load_candidates(
            "/private/home/belindali/temp/BLINK-Internal/models/entity_debug.jsonl",
            "/private/home/belindali/temp/BLINK-Internal/models/entity_encode_debug.t7",
            "/private/home/belindali/temp/BLINK-Internal/models/entity_ids_debug.t7",  # TODO MAKE THIS FILE!!!!
            biencoder, biencoder_params["max_cand_length"], args.entity_catalogue == args.test_entities, logger=logger
        )
    else:
        (
            candidate_encoding,
            candidate_token_ids,
        ) = _load_candidates(
            args.entity_catalogue, args.entity_encoding, args.entity_token_ids,
            biencoder, biencoder_params["max_cand_length"], args.entity_catalogue == args.test_entities,
            logger=logger
        )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        candidate_token_ids,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    candidate_encoding,
    candidate_token_ids,
):
    logger.info("Loading id2title")
    id2title = json.load(open("models/id2title.json"))
    logger.info("Finish loading id2title")

    if not args.test_mentions and not args.interactive and not args.qa_data:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entities (--test_entities)"
        )
        raise ValueError(msg)
    
    if getattr(args, 'save_preds_dir', None) is not None and not os.path.exists(args.save_preds_dir):
        os.makedirs(args.save_preds_dir)
        print("Saving preds in {}".format(args.save_preds_dir))

    print(args)
    print(args.output_path)

    stopping_condition = False
    if args.interactive:
        while not stopping_condition:

            logger.info("interactive mode")

            # Interactive
            text = input("insert text: ")

            # Prepare data
            samples = [{"id": "-1", "text": text}]
            dataloader = _process_biencoder_dataloader(
                samples, biencoder.tokenizer, biencoder_params, logger,
            )

            # Run inference
            nns, dists, pred_mention_bounds, mention_scores, cand_scores = _run_biencoder(
                args, biencoder, dataloader, candidate_encoding, samples=samples,
                top_k=args.top_k, device="cpu" if biencoder_params["no_cuda"] else "cuda",
                jointly_extract_mentions=("joint" in args.do_ner),
                # num_mentions=int(args.mention_classifier_threshold) if args.do_ner == "joint" else None,
                mention_classifier_threshold=float(args.mention_classifier_threshold) if "joint" in args.do_ner else None,
                # cand_encs_flat_index=cand_encs_flat_index
            )

            (
                all_entity_preds, num_correct, num_predicted, num_gold,
                num_correct_from_input_window, num_gold_from_input_window,
            ) = get_predictions(
                args, dataloader, biencoder_params,
                samples, nns, dists, mention_scores, cand_scores,
                pred_mention_bounds, id2title
            )

            print(samples[0]['text'])
            print("\n".join([
                entity[1] + "\n    Title: " + entity[0] + "\n    Score: " + str(all_entity_preds[0]['scores'][idx]) + "\n    Tuple: " + str(all_entity_preds[0]['pred_triples'][idx])
                for idx, entity in enumerate(all_entity_preds[0]['pred_tuples_string'])
            ]))
    
    else:
        samples = None

        logger.info("Loading test samples....")
        samples, num_unk = _get_test_samples(
            args.test_mentions, args.test_entities, logger,
            qa_data=True, do_ner=args.do_ner, debug=args.debug_biencoder,
        )
        logger.info("Finished loading test samples")
        logger.info("Preparing data for biencoder....")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params, logger,
        )
        logger.info("Finished preparing data for biencoder")

        stopping_condition = True

        # prepare the data for biencoder
        # run biencoder if predictions not saved
        if not os.path.exists(os.path.join(args.save_preds_dir, 'biencoder_mention_bounds.npy')):

            # run biencoder
            logger.info("Running biencoder...")

            start_time = time.time()
            nns, dists, pred_mention_bounds, mention_scores, cand_scores = _run_biencoder(
                args, biencoder, dataloader, candidate_encoding, samples=samples,
                top_k=args.top_k, device="cpu" if biencoder_params["no_cuda"] else "cuda",
                jointly_extract_mentions=("joint" in args.do_ner),
                # num_mentions=int(args.mention_classifier_threshold) if args.do_ner == "joint" else None,
                mention_classifier_threshold=float(args.mention_classifier_threshold) if "joint" in args.do_ner else None,
                # cand_encs_flat_index=cand_encs_flat_index
            )
            end_time = time.time()
            logger.info("Finished running biencoder")

            runtime = end_time - start_time
            
            np.save(os.path.join(args.save_preds_dir, "biencoder_nns.npy"), nns)
            np.save(os.path.join(args.save_preds_dir, "biencoder_dists.npy"), dists)
            np.save(os.path.join(args.save_preds_dir, "biencoder_mention_bounds.npy"), pred_mention_bounds)
            np.save(os.path.join(args.save_preds_dir, "biencoder_cand_scores.npy"), cand_scores)
            np.save(os.path.join(args.save_preds_dir, "biencoder_mention_scores.npy"), mention_scores)
            # json.dump(sample_to_all_context_inputs, open(os.path.join(args.save_preds_dir, "sample_to_all_context_inputs.json"), "w"))
            with open(os.path.join(args.save_preds_dir, "runtime.txt"), "w") as wf:
                wf.write(str(runtime))
        else:
            nns, dists, pred_mention_bounds, cand_scores, mention_scores = _retrieve_from_saved_biencoder_outs(args.save_preds_dir)
            runtime = float(open(os.path.join(args.save_preds_dir, "runtime.txt")).read())

        assert len(samples) == len(nns) == len(dists) == len(pred_mention_bounds) == len(cand_scores) == len(mention_scores)

        (
            all_entity_preds, num_correct, num_predicted, num_gold,
            num_correct_from_input_window, num_gold_from_input_window,
        ) = get_predictions(
            args, dataloader, biencoder_params,
            samples, nns, dists, mention_scores, cand_scores,
            pred_mention_bounds, id2title
        )
        
        print()
        if num_predicted > 0 and num_gold > 0:
            p = float(num_correct) / float(num_predicted)
            r = float(num_correct) / float(num_gold)
            p_window = float(num_correct_from_input_window) / float(num_predicted)
            r_window = float(num_correct_from_input_window) / float(num_gold_from_input_window)
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0
            if p_window + r_window > 0:
                f1_window = 2 * p_window * r_window / (p_window + r_window)
            else:
                f1_window = 0
            print("biencoder precision = {} / {} = {}".format(num_correct, num_predicted, p))
            print("biencoder recall = {} / {} = {}".format(num_correct, num_gold, r))
            print("biencoder f1 = {}".format(f1))
            print("Just entities within input window...")
            print("biencoder precision = {} / {} = {}".format(num_correct_from_input_window, num_predicted, p_window))
            print("biencoder recall = {} / {} = {}".format(num_correct_from_input_window, num_gold_from_input_window, r_window))
            print("biencoder f1 = {}".format(f1_window))
            print("*--------*")
            print("biencoder runtime = {}".format(runtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug", "-d", action="store_true", default=False, help="Run debug mode"
    )
    parser.add_argument(
        "--debug_biencoder", "-db", action="store_true", default=False, help="Debug biencoder"
    )
    # evaluation mode
    parser.add_argument(
        "--get_predictions", "-p", action="store_true", default=False, help="Getting predictions mode. Does not filter at crossencoder step."
    )
    parser.add_argument(
        "--eval_main_entity", action="store_true", default=False, help="Main-entity evaluation."
    )
    
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    parser.add_argument(
        "--qa_data", "-q", action="store_true", help="Test Data is QA form"
    )
    parser.add_argument(
        "--save_preds_dir", type=str, help="Directory to save model predictions to."
    )
    parser.add_argument(
        "--do_ner", "-n", type=str, default='none', choices=['joint_all_ents', 'joint', 'flair', 'ngram', 'single', 'qa_classifier', 'none'],
        help="Use automatic NER systems. Options: 'joint_all_ents', 'joint', 'flair', 'ngram', 'single', 'qa_classifier' 'none'."
        "(Set 'none' to get gold mention bounds from examples)"
    )
    parser.add_argument(
        "--mention_classifier_threshold", type=str, default=None, help="Must be specified if '--do_ner qa_classifier'."
        "Threshold for mention classifier score (either qa or joint) for which examples will be pruned if they fall under that threshold."
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Must be specified if '--do_ner qa_classifier'."
        "Number of entity candidates to consider per mention"
    )
    parser.add_argument(
        "--final_thresholding", type=str, default=None, help="How to threshold the final candidates."
        "`top_joint_by_mention`: get top candidate (with joint score) for each predicted mention bound."
        "`top_entity_by_mention`: get top candidate (with entity score) for each predicted mention bound."
        "`joint_0`: by thresholding joint score to > 0."
    )


    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        # default="models/biencoder_wiki_large.bin",
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        # default="models/biencoder_wiki_large.json",
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_token_ids",
        dest="entity_token_ids",
        type=str,
        default="models/entity_token_ids_128.t7",  # ALL WIKIPEDIA!
        help="Path to the tokenized entity titles + descriptions.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=8,
        help="Crossencoder's batch size for evaluation",
    )
    parser.add_argument(
        "--max_context_length",
        dest="max_context_length",
        type=int,
        help="Maximum length of context. (Don't set to inherit from training config)",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--use_cuda", dest="use_cuda", action="store_true", default=False, help="run on gpu"
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)
    logger.setLevel(10)

    models = load_models(args, logger)
    run(args, logger, *models)
