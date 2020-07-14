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


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


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
    tensor_data = TensorDataset(*tensor_data_tuple)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(
    args, biencoder, dataloader, candidate_encoding, samples,
    top_k=100, device="cpu", jointly_extract_mentions=False,
    sample_to_all_context_inputs=None, num_mentions=10,  # TODO don't hardcode
    mention_classifier_threshold=0.25,
    cand_encs_flat_index=None,
):
    """
    Returns: tuple
        labels (List[int]) [(max_num_mentions_gold) x exs]: gold labels
        nns (List[Array[int]]) [(# of pred mentions, cands_per_mention) x exs]: predicted entity IDs in each example
        dists (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: scores of each entity in nns
        pred_mention_bounds (List[Array[int]]) [(# of pred mentions, 2) x exs]: predicted mention boundaries in each examples
        cand_dists (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: candidate score component (- mention) score
    """
    biencoder.model.eval()
    labels = []
    context_inputs = []
    nns = []
    dists = []
    mention_dists = []
    pred_mention_bounds = []
    cand_dists = []
    sample_idx = 0
    ctxt_idx = 0
    new_samples = samples
    new_sample_to_all_context_inputs = sample_to_all_context_inputs
    if jointly_extract_mentions:
        new_samples = []
        new_sample_to_all_context_inputs = []
    for step, batch in enumerate(tqdm(dataloader)):
        context_input, _, label_ids, mention_idxs, mention_idxs_mask = batch
        with torch.no_grad():
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(context_input, biencoder.NULL_IDX)
            context_encoding, _, _ = biencoder.model.context_encoder.bert_model(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,  # what is segment IDs?
            )
            mention_logits, mention_bounds = biencoder.model.classification_heads['mention_scores'](context_encoding, mask_ctxt)

            '''
            topK_mention_scores, mention_pos = torch.cat([torch.arange(), mention_logits.topk(top_k, dim=1)])
            mention_pos = mention_pos.flatten()
            '''
            # DIM (num_total_mentions, embed_dim)
            # mention_pos = (torch.sigmoid(mention_logits) >= mention_classifier_threshold).nonzero()
            # import pdb
            # pdb.set_trace()
            # start_time = time.time()
            # mention_pos = (torch.sigmoid(mention_logits) >= 0.2).nonzero()
            # end_time = time.time()
            # (bsz, top_k); (bsz, top_k)
            top_mention_logits, mention_pos_2 = mention_logits.topk(top_k)
            # 2nd part of OR for if nothing is > 0
            mention_pos_2 = torch.stack([torch.arange(mention_pos_2.size(0)).unsqueeze(-1).expand_as(mention_pos_2), mention_pos_2], dim=-1)
            mention_pos_2_mask = torch.sigmoid(top_mention_logits) >= mention_classifier_threshold
            # [overall mentions, 2]
            mention_pos_2 = mention_pos_2[mention_pos_2_mask | (mention_pos_2_mask.sum(1) == 0).unsqueeze(-1)]
            mention_pos_2 = mention_pos_2.view(-1, 2)
            # end_time_2 = time.time()
            # print(end_time - start_time)
            # print(end_time_2 - end_time)
            # tuples of (instance in batch, mention id) of what to include
            mention_pos = mention_pos_2
            # TODO MAYBE TOP K HERE??
            # '''

            # mention_pos_mask = torch.sigmoid(mention_logits) > 0.25
            # reshape back to (bs, num_mentions) mask
            mention_pos_mask = torch.zeros(mention_logits.size(), dtype=torch.bool).to(mention_pos.device)
            mention_pos_mask[mention_pos_2[:,0], mention_pos_2[:,1]] = 1
            # (bs, num_mentions, 2)
            mention_idxs = mention_bounds.clone()
            mention_idxs[~mention_pos_mask] = 0

            # take highest scoring mention
            # (bs, num_mentions, embed_dim)
            embedding_ctxt = biencoder.model.classification_heads['get_context_embeds'](context_encoding, mention_idxs)
            if biencoder.model.linear_compression is not None:
                embedding_ctxt = biencoder.model.linear_compression(embedding_ctxt)

            # (num_total_mentions, embed_dim)
            embedding_ctxt = embedding_ctxt[mention_pos_mask]
            # (num_total_mentions, 2)
            mention_idxs = mention_idxs[mention_pos_mask]

            # DIM (num_total_mentions, num_candidates)
            # TODO search for topK entities with FAISS
            # start_time = time.time()
            if embedding_ctxt.size(0) > 1:
                cand_scores = embedding_ctxt.squeeze(0).mm(candidate_encoding.to(device).t())
            else:
                cand_scores = embedding_ctxt.mm(candidate_encoding.to(device).t())
            # end_time = time.time()
            # cand_scores = torch.log_softmax(cand_scores, 1)
            cand_dist, cand_indices = cand_scores.topk(10)  # TODO DELETE
            cand_scores = torch.log_softmax(cand_dist, 1)
            # back into (num_total_mentions, num_candidates)
            # cand_scores_reconstruct = torch.ones(embedding_ctxt.size(0), candidate_encoding.size(0), dtype=cand_scores.dtype).to(cand_scores.device) * -float("inf")
            # # # DIM (bs, max_pred_mentions, num_candidates)
            # cand_scores_reconstruct[torch.arange(cand_scores_reconstruct.size(0)).unsqueeze(-1), cand_indices] = cand_scores
            # reconstruct_time = time.time()
            # cand_scores = cand_scores_reconstruct
            # print(top_time - softmax_time)
            # print(soft_time - top_time)
            # print(reconstruct_time - soft_time)
            # reconstruct_time = time.time()
            # print(softmax_time - end_time)
            # print(reconstruct_time - softmax_time)

            # scores = F.log_softmax(cand_scores, dim=-1)
            # softmax_time = time.time()
            # cand_scores, _ = cand_encs_flat_index.search(embedding_ctxt.contiguous().detach().cpu().numpy(), top_k)
            # cand_scores = torch.tensor(cand_scores)
            # # reconstruct cand_scores to (bs, num_candidates)
            # faiss_time = time.time()
            # print(end_time - start_time)
            # print(softmax_time - end_time)
            # print(faiss_time - softmax_time)
            # DIM (num_total_mentions, num_candidates)
            if args.final_thresholding != "top_entity_by_mention":
                # DIM (num_total_mentions, num_candidates)
                # log p(entity && mb) = log [p(entity|mention bounds) * p(mention bounds)] = log p(e|mb) + log p(mb)
                # scores += torch.sigmoid(mention_logits)[mention_pos_mask].unsqueeze(-1)
                mention_scores = mention_logits[mention_pos_mask].unsqueeze(-1)
            # DIM (num_total_mentions, num_candidates)
            # scores = torch.log_softmax(cand_scores, 1) + torch.sigmoid(mention_scores)
            scores = cand_scores + torch.sigmoid(mention_scores)
            mention_scores = mention_scores.expand_as(cand_scores)

            # # # DIM (bs, max_pred_mentions, num_candidates)
            # scores_reconstruct = torch.zeros(mention_pos_mask.size(0), mention_pos_mask.sum(1).max(), scores.size(-1), dtype=scores.dtype).to(scores.device)
            # # DIM (bs, max_pred_mentions)
            mention_pos_mask_reconstruct = torch.zeros(mention_pos_mask.size(0), mention_pos_mask.sum(1).max()).bool().to(mention_pos_mask.device)
            for i in range(mention_pos_mask_reconstruct.size(1)): mention_pos_mask_reconstruct[:, i] = i < mention_pos_mask.sum(1)
            # # # DIM (bs, max_pred_mentions, num_candidates)
            # scores_reconstruct[mention_pos_mask_reconstruct] = scores
            # scores = scores_reconstruct

            # # DIM (bs, max_pred_mentions, num_candidates)
            # cand_scores_reconstruct = torch.zeros(mention_pos_mask.size(0), mention_pos_mask.sum(1).max(), scores.size(-1), dtype=scores.dtype).to(scores.device)
            # # DIM (bs, max_pred_mentions, num_candidates)
            # cand_scores_reconstruct[mention_pos_mask_reconstruct] = cand_scores
            # cand_scores = cand_scores_reconstruct

            # DIM (bs, max_pred_mentions, 2)
            chosen_mention_bounds = torch.zeros(mention_pos_mask.size(0), mention_pos_mask.sum(1).max(), 2, dtype=mention_idxs.dtype).to(mention_idxs.device)
            chosen_mention_bounds[mention_pos_mask_reconstruct] = mention_idxs

            # mention_idxs = mention_idxs.view(topK_mention_bounds.size(0), topK_mention_bounds.size(1), 2)
            # assert (mention_idxs == topK_mention_bounds).all()

            # DIM (total_num_mentions, num_cands)
            # scores = scores[mention_pos_mask_reconstruct]
            # cand_scores = cand_scores[mention_pos_mask_reconstruct]
            # mention_scores = scores - 0.4 * cand_scores - 0.9  # project to dimension of scores

            # expand labels
            # DIM (total_num_mentions, 1)
            # label_ids = label_ids.expand(chosen_mention_bounds.size(0), chosen_mention_bounds.size(1))[mention_pos_mask_reconstruct].unsqueeze(-1)

        # for i, instance in enumerate(chosen_mention_bounds):
        #     new_sample_to_all_context_inputs.append([])
        #     # if len(chosen_mention_bounds[i][mention_pos_mask_reconstruct[i]]) == 0:
        #     #     new_samples.append({})
        #     #     for key in samples[sample_idx]:
        #     #         if key != "context_left" and key != "context_right" and key != "mention":
        #     #             new_samples[ctxt_idx][key] = samples[sample_idx][key]
        #     for j, mention_bound in enumerate(chosen_mention_bounds[i][mention_pos_mask_reconstruct[i]]):
        #         new_sample_to_all_context_inputs[sample_idx].append(ctxt_idx)
        #         context_left = _decode_tokens(biencoder.tokenizer, context_input[i].tolist()[:mention_bound[0]]) + " "
        #         context_right = " " + _decode_tokens(biencoder.tokenizer, context_input[i].tolist()[mention_bound[1] + 1:])  # mention bound is inclusive
        #         mention = _decode_tokens(
        #             biencoder.tokenizer,
        #             context_input[i].tolist()[mention_bound[0]:mention_bound[1] + 1]
        #         )
        #         new_samples.append({
        #             "context_left": context_left,
        #             "context_right": context_right,
        #             "mention": mention,
        #         })
        #         for key in samples[sample_idx]:
        #             if key != "context_left" and key != "context_right" and key != "mention":
        #                 new_samples[ctxt_idx][key] = samples[sample_idx][key]
        #         ctxt_idx += 1
        #     sample_idx += 1

        dist, indices = scores.sort(descending=True)
        indices = torch.gather(cand_indices, -1, indices)
        # Do all reconstructions
        # (bsz, max_num_mentions=top_k, cands_per_mention)
        indices_reconstruct = -torch.ones(mention_pos_mask.size(0), mention_pos_mask.sum(1).max(), indices.size(-1), dtype=indices.dtype).to(indices.device)
        dist_reconstruct = indices_reconstruct.clone().float()
        cand_dist_reconstruct = dist_reconstruct.clone()
        indices_reconstruct[mention_pos_mask_reconstruct] = indices
        dist_reconstruct[mention_pos_mask_reconstruct] = dist
        cand_dist_reconstruct[mention_pos_mask_reconstruct] = cand_dist
        indices = indices_reconstruct
        dist = dist_reconstruct
        cand_dist = cand_dist_reconstruct
        
        # [(max_num_mentions_gold) x exs] <= (bsz, max_num_mentions_gold)
        labels.extend(label_ids.data.numpy())
        # [(seqlen) x exs] <= (bsz, seqlen)
        context_inputs.extend(context_input.data.numpy())
        # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
        nns.extend(indices.data.cpu().numpy())
        # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
        dists.extend(dist.data.cpu().numpy())
        # [(max_num_mentions, 2) x exs] <= (bsz, max_num_mentions=top_k, 2)
        pred_mention_bounds.extend(chosen_mention_bounds.data.cpu().numpy())
        # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=top_k, cands_per_mention)
        cand_dists.extend(cand_dist.data.cpu().numpy())
        assert len(labels) == len(nns)
        assert len(labels) == len(dists)
        sys.stdout.write("{}/{} \r".format(step, len(dataloader)))
        sys.stdout.flush()
    # if jointly_extract_mentions:
    #     assert sample_idx == len(new_sample_to_all_context_inputs)
    #     assert ctxt_idx == len(new_samples)
    return labels, nns, dists, pred_mention_bounds, cand_dists  #, new_samples, new_sample_to_all_context_inputs


def _decode_tokens(tokenizer, token_list):
    decoded_string = tokenizer.decode(token_list)
    if isinstance(decoded_string, list):
        decoded_string = decoded_string[0] if len(decoded_string) > 0 else ""
    if "[CLS]" in decoded_string:
        decoded_string = decoded_string[len('[CLS] '):].strip()  # disrgard CLS token
    return decoded_string


def _retrieve_from_saved_biencoder_outs(save_preds_dir):
    labels = np.load(os.path.join(args.save_preds_dir, "biencoder_labels.npy"))
    nns = np.load(os.path.join(args.save_preds_dir, "biencoder_nns.npy"))
    dists = np.load(os.path.join(args.save_preds_dir, "biencoder_dists.npy"))
    pred_mention_bounds = np.load(os.path.join(args.save_preds_dir, "biencoder_mention_bounds.npy"))
    return labels, nns, dists, pred_mention_bounds


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

    if not args.test_mentions and not args.interactive and not args.qa_data:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entities (--test_entities)"
        )
        raise ValueError(msg)
    
    if hasattr(args, 'save_preds_dir') and not os.path.exists(args.save_preds_dir):
        os.makedirs(args.save_preds_dir)
        print("Saving preds in {}".format(args.save_preds_dir))

    print(args)
    print(args.output_path)

    stopping_condition = False
    while not stopping_condition:

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
            labels, nns, dists, pred_mention_bounds, cand_dists = _run_biencoder(
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
            
            np.save(os.path.join(args.save_preds_dir, "biencoder_labels.npy"), labels)
            np.save(os.path.join(args.save_preds_dir, "biencoder_nns.npy"), nns)
            np.save(os.path.join(args.save_preds_dir, "biencoder_dists.npy"), dists)
            np.save(os.path.join(args.save_preds_dir, "biencoder_mention_bounds.npy"), pred_mention_bounds)
            np.save(os.path.join(args.save_preds_dir, "biencoder_cand_dists.npy"), cand_dists)
            # json.dump(sample_to_all_context_inputs, open(os.path.join(args.save_preds_dir, "sample_to_all_context_inputs.json"), "w"))
            with open(os.path.join(args.save_preds_dir, "runtime.txt"), "w") as wf:
                wf.write(str(runtime))
        else:
            labels, nns, dists, pred_mention_bounds = _retrieve_from_saved_biencoder_outs(args.save_preds_dir)
            runtime = float(open(os.path.join(args.save_preds_dir, "runtime.txt")).read())

        # assert len(samples) == len(labels) == len(nns) == len(dists)

        # save biencoder predictions and print precision/recalls
        num_correct = 0
        num_predicted = 0
        num_gold = 0
        num_correct_from_input_window = 0
        num_gold_from_input_window = 0
        save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
        all_entity_preds = []
        # check out no_pred_indices
        with open(save_biencoder_file, 'w') as f:

            # labels (List[int]) [(max_num_mentions_gold) x exs]
            # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
            # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
            # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
            # cand_dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
            for batch_num, batch_data in enumerate(dataloader):
                batch_context, batch_cands, batch_label_ids, batch_mention_idxs, batch_mention_idx_masks = batch_data
                for b in range(biencoder_params['eval_batch_size']):
                    i = batch_num * biencoder_params['eval_batch_size'] + b
                    sample = samples[i]
                    input_mention_idxs = batch_mention_idxs[b][batch_mention_idx_masks[b]].tolist()
                    input_label_ids = batch_label_ids[b][batch_label_ids[b] != -1].tolist()
                    assert len(input_label_ids) == len(input_mention_idxs)

                    # (num_pred_mentions, cands_per_mention)
                    pred_entity_list = nns[i][nns[i][:,0] != -1]
                    if len(pred_entity_list) > 0:
                        e_id = pred_entity_list[0]
                    distances = dists[i][nns[i][:,0] != -1]
                    # (num_pred_mentions, 2)
                    entity_mention_bounds_idx = pred_mention_bounds[i][nns[i][:,0] != -1]
                    # (max_num_mentions_gold)
                    label = labels[i][labels[i] != -1]
                    utterance = sample['text']
                    gold_mention_bounds = [
                        sample['text'][ment[0]-10:ment[0]] + "[" + sample['text'][ment[0]:ment[1]] + "]" + sample['text'][ment[1]:ment[1]+10]
                        for ment in sample['mentions']
                    ]

                    '''
                    get top for each mention bound, w/out duplicates
                    # TOP-1
                    all_pred_entities = pred_entity_list[:,:1]
                    e_mention_bounds = entity_mention_bounds_idx[:1].tolist()
                    # '''
                    if args.final_thresholding == "joint_0":
                        # THRESHOLDING
                        assert utterance is not None
                        top_mentions = (distances[:,0] > 0).nonzero()[0]
                        # cands already sorted by score
                        all_pred_entities = [pred_entity_list[ment_idx, 0] for ment_idx in top_mentions]
                        e_mention_bounds = [entity_mention_bounds_idx[ment_idx] for ment_idx in top_mentions]
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
                    mention_masked_utterance = np.zeros(len(sample['tokenized_text_ids']))
                    # ensure well-formed-ness, prune overlaps
                    # greedily pick highest scoring, then prune all overlapping
                    for idx, mb in enumerate(e_mention_bounds):
                        mb[1] += 1  # prediction was inclusive, now make exclusive
                        # check if in existing mentions
                        try:
                            if mention_masked_utterance[mb[0]] == 1 or mention_masked_utterance[mb[1] - 1] == 1:
                                continue
                        except:
                            import pdb
                            pdb.set_trace()
                        e_mention_bounds_pruned.append(mb)
                        all_pred_entities_pruned.append(all_pred_entities[idx])
                        mention_masked_utterance[mb[0]:mb[1]] = 1

                    # GET ALIGNED MENTION_IDXS (input is slightly different to model) between ours and gold labels
                    gold_input = sample['tokenized_text_ids']
                    my_input = batch_context[b][1:-1].tolist()  # remove bos and sep
                    # return first instance of my_input in gold_input
                    for my_input_start in range(len(gold_input)):
                        if gold_input[my_input_start] == my_input[0] and gold_input[my_input_start:my_input_start+len(my_input)] == my_input:
                            break
                    my_input_start -= 1  # for bos

                    # add alignment factor (my_input_start) to predicted mention triples
                    pred_triples = [(
                        # sample['all_gold_entities'][i],
                        str(all_pred_entities_pruned[j]),
                        int(e_mention_bounds_pruned[j][0] + my_input_start),
                        int(e_mention_bounds_pruned[j][1] + my_input_start),
                    ) for j in range(len(all_pred_entities_pruned))]
                    gold_triples = [(
                        str(sample['label_id'][j]),
                        sample['tokenized_mention_idxs'][j][0], sample['tokenized_mention_idxs'][j][1],
                    ) for j in range(len(sample['label_id']))]
                    num_correct += entity_linking_tp_with_overlap(gold_triples, pred_triples)
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

                    # if sample['label'] is not None:
                    #     num_predicted += 1
                    entity_results = {
                        "id": sample["id"],
                        # "top_ent_id": e_id,
                        # "all_gold_entities": sample.get("label_id", None),
                        # "id": e_id,
                        # "title": e_title,
                        # "text": e_text,
                        "pred_triples": pred_triples,
                        "gold_triples": gold_triples,
                        # "pred_mention_bounds": pred_mention_bounds,
                        # "gold_mention_bounds": gold_mention_bounds,
                        # "gold_ent_id": sample['label_id'],
                        "scores": distances.tolist(),
                    }

                    all_entity_preds.append(entity_results)
                    f.write(
                        json.dumps(entity_results) + "\n"
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
        "--top_k", type=int, default=100, help="Must be specified if '--do_ner qa_classifier'."
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
