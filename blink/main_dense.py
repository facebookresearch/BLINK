# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_context_representation,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.train_cross import modify, evaluate
import math

import vcg_utils
from vcg_utils.mention_extraction import extract_entities
from vcg_utils.measures import entity_linking_tp_with_overlap

import os
import sys
from tqdm import tqdm
import pdb


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


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
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    kb2id = {}
    id2kb = {}
    resave_encodings = False
    bsz = 128
    candidate_rep = []

    wikipedia_id2local_id = {}
    local_idx = 0
    missing_entity_ids = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for i, line in enumerate(tqdm(lines)):
            entity = json.loads(line)
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            if get_kbids:
                if "kb_idx" in entity:
                    kb2id[entity["kb_idx"]] = local_idx
                    id2kb[local_idx] = entity["kb_idx"]
                else:
                    missing_entity_ids += 1
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1

            if i > len(candidate_encoding):
                resave_encodings = True
                # not in candidate encodings file, generate through forward method
                candidate_rep.append(get_candidate_representation(
                    # entity["title"] + " "
                    entity["text"].strip(), biencoder.tokenizer,
                    128, entity["title"].strip()
                )['ids'])
                if len(candidate_rep) == bsz or i == len(lines) - 1:
                    try:
                        curr_cand_encs = biencoder.encode_candidate(
                            torch.LongTensor(candidate_rep)
                        )
                        with open("models/entities_with_ids.txt", "a") as f:
                            d=f.write(json.dumps(curr_cand_encs.tolist()) + "\n")
                    except RuntimeError:
                        import pdb
                        pdb.set_trace()
                    candidate_rep = []

    if resave_encodings:
        torch.save(candidate_encoding, "new_" + entity_encoding)
    if logger:
        logger.info("missing {}/{} wikidata IDs".format(missing_entity_ids, local_idx))
    return candidate_encoding, candidate_token_ids, title2id, id2title, id2text, wikipedia_id2local_id, kb2id, id2kb


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    id2kb = {}
    missing_pages = 0
    missing_entity_ids = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                if "kb_idx" in entity:
                    kb2id[entity["kb_idx"]] = title2id[entity["title"]]
                    id2kb[title2id[entity["title"]]] = entity["kb_idx"]
                else:
                    missing_entity_ids += 1
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
        logger.info("missing {}/{} wikidata IDs".format(missing_entity_ids, n))
    return kb2id, id2kb


def get_mention_bound_candidates(
    do_ner, record, new_record,
    saved_ngrams=None, ner_model=None, max_mention_len=4,
    ner_errors=0, sample_idx=0, qa_classifier_saved=None,
):
    if do_ner == "ngram":
        if record["utterance"] not in saved_ngrams:
            tagged_text = vcg_utils.utils.get_tagged_from_server(record["utterance"], caseless=True)
            # get annotated entities for each sample
            samples = []
            for mention_len in range(min(len(tagged_text), max_mention_len), 0, -1):
                annotated_sample = extract_entities(tagged_text, ngram_len=mention_len)
                samples += annotated_sample
            saved_ngrams[record["utterance"]] = samples
        else:
            samples = saved_ngrams[record["utterance"]]
        # setup new_record
        for sample in samples:
            sample["context_left"] = record['utterance'][:sample['offsets'][0]]
            sample["context_right"] = record['utterance'][sample['offsets'][1]:]
            sample["mention"] = record['utterance'][sample['offsets'][0]:sample['offsets'][1]]

    elif do_ner == "flair":
        # capitalize each word
        sentences = [record["utterance"].title()]
        samples = _annotate(ner_model, sentences)
        if len(samples) == 0:
            ner_errors += 1
            # sample_to_all_context_inputs.append([])
            return None, None, saved_ngrams, sample_idx

    elif do_ner == "single":
        # assume single mention boundary
        samples = [{
            "context_left": "",
            "context_right": "",
            "mention": record["utterance"],
        }]

    elif do_ner == "qa_classifier":
        samples = []
        for _q in qa_classifier_saved[record['question_id']]:
            # TODO read from file
            samples.append({
                "context_left": _q['passage'].split('<answer>')[0],
                "context_right": _q['passage'].split('</answer>')[1],
                "mention": _q['text'],
            })

    new_record_list = []
    sample_idx_list = []
    if do_ner != "none":
        for sample in samples:
            new_record_list.append({
                "q_id": new_record["q_id"],
                "label": new_record["label"],
                "label_id": new_record["label_id"],
                "context_left": sample["context_left"].lower(),
                "context_right": sample["context_right"].lower(),
                "mention": sample["mention"].lower(),
            })
            if "all_gold_entities" in new_record:
                new_record_list[len(new_record_list)-1]["all_gold_entities"] = new_record["all_gold_entities"]
                new_record_list[len(new_record_list)-1]["all_gold_entities_ids"] = new_record["all_gold_entities_ids"]
            if "main_entity_pos" in record:
                new_record_list[len(new_record_list)-1]["gold_context_left"] = record[
                    'utterance'][:record['main_entity_pos'][0]].lower()
                new_record_list[len(new_record_list)-1]["gold_context_right"] = record[
                    'utterance'][record['main_entity_pos'][1]:].lower()
                new_record_list[len(new_record_list)-1]["gold_mention"] = record[
                    'utterance'][record['main_entity_pos'][0]:record['main_entity_pos'][1]].lower()
            sample_idx_list.append(sample_idx)
            sample_idx += 1
        if len(samples) == 0:
            # found no samples for this record...
            new_record_list.append({
                "q_id": new_record["q_id"],
                "label": new_record["label"],
                "label_id": new_record["label_id"],
                "context_left": "", "context_right": "", "mention": "",
            })
            if "all_gold_entities" in new_record:
                new_record_list[len(new_record_list)-1]["all_gold_entities"] = new_record["all_gold_entities"]
                new_record_list[len(new_record_list)-1]["all_gold_entities_ids"] = new_record["all_gold_entities_ids"]
            if "main_entity_pos" in record:
                new_record_list[len(new_record_list)-1]["gold_context_left"] = record[
                    'utterance'][:record['main_entity_pos'][0]].lower()
                new_record_list[len(new_record_list)-1]["gold_context_right"] = record[
                    'utterance'][record['main_entity_pos'][1]:].lower()
                new_record_list[len(new_record_list)-1]["gold_mention"] = record[
                    'utterance'][record['main_entity_pos'][0]:record['main_entity_pos'][1]].lower()
            sample_idx_list.append(sample_idx)
            sample_idx += 1
    else:
        # entity bounds are given
        entity_bounds = record["main_entity_pos"]
        new_record["context_left"] = record["utterance"][:entity_bounds[0]].lower()
        new_record["gold_context_left"] = new_record["context_left"].lower()
        new_record["context_right"] = record["utterance"][entity_bounds[1]:].lower()
        new_record["gold_context_right"] = new_record["context_right"].lower()
        new_record["mention"] = record["main_entity_tokens"].lower()
        new_record["gold_meniton"] = new_record["mention"].lower()
        if "all_gold_entities" in new_record:
            new_record["all_gold_entities"] = new_record["all_gold_entities"]
            new_record["all_gold_entities_ids"] = new_record["all_gold_entities_ids"]
        new_record_list = [new_record]
        sample_idx_list = [sample_idx]
        sample_idx += 1

    return new_record_list, sample_idx_list, saved_ngrams, ner_errors, sample_idx


def __load_test(
    test_filename, kb2id, logger, args,
    qa_data=False, id2kb=None, title2id=None,
    do_ner="none", use_ngram_extractor=False, max_mention_len=4,
    debug=False, main_entity_only=False,
):
    test_samples = []
    sample_to_all_context_inputs = []  # if multiple mentions found for an example, will have multiple inputs
                                        # maps indices of examples to list of all indices in `samples`
                                        # i.e. [[0], [1], [2, 3], ...]
    unknown_entity_samples = []
    num_unknown_entity_samples = 0
    num_no_gold_entity = 0
    ner_errors = 0
    ner_model = None
    if do_ner == "flair":
        # Load NER model
        ner_model = NER.get_model()

    saved_ngrams = {}
    if do_ner == "ngram":
        save_file = "{}_saved_ngrams_new_rules_{}.json".format(test_filename, max_mention_len)
        if os.path.exists(save_file):
            saved_ngrams = json.load(open(save_file))

    qa_classifier_saved = {}
    if do_ner == "qa_classifier":
        assert getattr(args, 'qa_classifier_threshold', None) is not None
        if args.qa_classifier_threshold == "top1":
            do_top_1 = True
        else:
            do_top_1 = False
            qa_classifier_threshold = float(args.qa_classifier_threshold)
        if "webqsp.test" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/webqsp/test_predictions.json"
        elif "webqsp.dev" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/webqsp/dev_predictions.json"
        elif "graph.test" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/graphquestions/test_predictions.json"
        with open(test_predictions_json) as f:
            for line in f:
                line_json = json.loads(line)
                all_ex_preds = []
                for i, pred in enumerate(line_json['all_predictions']):
                    if "test" in test_filename:
                        pred['logit'][1] = math.log(pred['logit'][1])
                    if (
                        (do_top_1 and i == 0) or 
                        (not do_top_1 and pred['logit'][1] > qa_classifier_threshold)
                        # or i == 0  # have at least 1 candidate
                    ):
                        all_ex_preds.append(pred)
                assert '1' in line_json['predictions']
                qa_classifier_saved[line_json['id']] = all_ex_preds

    with open(test_filename, "r") as fin:
        if qa_data:
            lines = json.load(fin)
            sample_idx = 0
            do_setup_samples = True

            for i, record in enumerate(tqdm(lines)):
                new_record = {}
                new_record["q_id"] = record["question_id"]

                if main_entity_only:
                    if "main_entity" not in record or record["main_entity"] is None:
                        num_no_gold_entity += 1
                        new_record["label"] = None
                        new_record["label_id"] = -1
                        new_record["all_gold_entities"] = []
                        new_record["all_gold_entities_ids"] = []
                    elif record['main_entity'] in kb2id:
                        new_record["label"] = record["main_entity"]
                        new_record["label_id"] = kb2id[record['main_entity']]
                        new_record["all_gold_entities"] = [record["main_entity"]]
                        new_record["all_gold_entities_ids"] = [kb2id[record['main_entity']]]
                    else:
                        num_unknown_entity_samples += 1
                        unknown_entity_samples.append(record)
                        # sample_to_all_context_inputs.append([])
                        # TODO DELETE?
                        continue
                else:
                    new_record["label"] = None
                    new_record["label_id"] = -1
                    if "entities" not in record or record["entities"] is None or len(record["entities"]) == 0:
                        if "main_entity" not in record or record["main_entity"] is None:
                            num_no_gold_entity += 1
                            new_record["all_gold_entities"] = []
                            new_record["all_gold_entities_ids"] = []
                        else:
                            new_record["all_gold_entities"] = [record["main_entity"]]
                            new_record["all_gold_entities_ids"] = [kb2id[record['main_entity']]]
                    else:
                        new_record["all_gold_entities"] = record['entities']
                        new_record["all_gold_entities_ids"] = []
                        for ent_id in new_record["all_gold_entities"]:
                            if ent_id in kb2id:
                                new_record["all_gold_entities_ids"].append(kb2id[ent_id])
                            else:
                                num_unknown_entity_samples += 1
                                unknown_entity_samples.append(record)

                (new_record_list, sample_idx_list,
                saved_ngrams, ner_errors, sample_idx) = get_mention_bound_candidates(
                    do_ner, record, new_record,
                    saved_ngrams=saved_ngrams, ner_model=ner_model,
                    max_mention_len=max_mention_len, ner_errors=ner_errors,
                    sample_idx=sample_idx, qa_classifier_saved=qa_classifier_saved,
                )
                if sample_idx_list is not None:
                    sample_to_all_context_inputs.append(sample_idx_list)
                if new_record_list is not None:
                    test_samples += new_record_list

        else:
            lines = fin.readlines()
            for line in tqdm(lines):
                record = json.loads(line)
                record["label"] = record["label_id"]
                record["q_id"] = record["query_id"]
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                    # LOWERCASE EVERYTHING !
                    record["context_left"] = record["context_left"].lower()
                    record["context_right"] = record["context_right"].lower()
                    record["mention"] = record["mention"].lower()
                    record["gold_context_left"] = record["context_left"].lower()
                    record["gold_context_right"] = record["context_right"].lower()
                    record["gold_mention"] = record["mention"].lower()
                    test_samples.append(record)

    # save info and log
    with open("saved_preds/unknown.json", "w") as f:
        json.dump(unknown_entity_samples, f)
    if do_ner == "ngram":
        save_file = "{}_saved_ngrams_new_rules_{}.json".format(test_filename, max_mention_len)
        with open(save_file, "w") as f:
            json.dump(saved_ngrams, f)
        logger.info("Finished saving to {}".format(save_file))
    if logger:
        logger.info("{}/{} samples considered".format(len(sample_to_all_context_inputs), len(lines)))
        logger.info("{} samples generated".format(len(test_samples)))
        logger.info("{} samples with unknown entities considered".format(num_unknown_entity_samples))
        logger.info("{} samples with no gold entities considered".format(num_no_gold_entity))
        logger.info("ner errors: {}".format(ner_errors))
    return test_samples, num_unknown_entity_samples, sample_to_all_context_inputs


def _get_test_samples(
    test_filename, test_entities_path, title2id, kb2id, id2kb, logger,
    qa_data=False, do_ner="none", debug=False, main_entity_only=False, do_map_test_entities=True,
):
    # TODO GET CORRECT IDS
    # if debug:
    #     test_entities_path = "/private/home/belindali/temp/BLINK-Internal/models/entity_debug.jsonl"
    if do_map_test_entities:
        kb2id, id2kb = __map_test_entities(test_entities_path, title2id, logger)
    test_samples, num_unk, sample_to_all_context_inputs = __load_test(
        test_filename, kb2id, logger, args,
        qa_data=qa_data, id2kb=id2kb, title2id=title2id,
        do_ner=do_ner, debug=debug, main_entity_only=main_entity_only,
    )
    return test_samples, kb2id, id2kb, num_unk, sample_to_all_context_inputs


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    tokens_data, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=False,
        logger=None,
        debug=biencoder_params["debug"],
        add_mention_bounds=(not args.no_mention_bounds_biencoder),
        get_cached_representation=False,  # TODO???
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, device="cpu"):
    biencoder.model.eval()
    labels = []
    context_inputs = []
    nns = []
    dists = []
    for step, batch in enumerate(tqdm(dataloader)):
        context_input, _, label_ids, mention_idxs = batch
        with torch.no_grad():
            scores = biencoder.score_candidate(
                context_input, None,
                cand_encs=candidate_encoding.to(device),
                mention_idxs=mention_idxs.to(device),
            )
        dist, indices = scores.topk(top_k)
        labels.extend(label_ids.data.numpy())
        context_inputs.extend(context_input.data.numpy())
        nns.extend(indices.data.cpu().numpy())
        dists.extend(dist.data.cpu().numpy())
        sys.stdout.write("{}/{} \r".format(step, len(dataloader)))
        sys.stdout.flush()
    return labels, nns, dists


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(
    crossencoder, dataloader, logger, context_len, device="cuda",
    id2kb=None, get_all_scores=False, forward_only=False
):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(
        crossencoder, dataloader, device, logger, context_len, silent=False, id2kb=id2kb,
        forward_only=forward_only)
    accuracy = res["normalized_accuracy"]
    return accuracy, res


def _combine_same_inputs_diff_mention_bounds(samples, labels, nns, dists, sample_to_all_context_inputs, filtered_indices=None, debug=False):
    # TODO save ALL samples
    if not debug:
        assert len(nns) == sample_to_all_context_inputs[-1][-1] + 1
    samples_merged = []
    nns_merged = []
    dists_merged = []
    labels_merged = []
    entity_mention_bounds_idx = []
    filtered_cluster_indices = []  # indices of entire chunks that are filtered out
    dists_idx = 0  # dists is already filtered, use separate idx to keep track of where we are
    for i, context_input_idxs in enumerate(sample_to_all_context_inputs):
        if debug:
            if context_input_idxs[0] >= len(nns):
                break
            elif context_input_idxs[-1] >= len(nns):
                context_input_idxs = context_input_idxs[:context_input_idxs.index(len(nns))]
        if len(context_input_idxs) == 0:
            # should not happen anymore...
            import pdb
            pdb.set_trace()
        # first filter all filetered_indices
        if filtered_indices is not None:
            context_input_idxs = [idx for idx in context_input_idxs if idx not in filtered_indices]
            # context_input_idxs_filt = []
            # for idx in context_input_idxs:
            #     if idx in filtered_indices:
            #         import pdb
            #         pdb.set_trace()
            #         num_filtered_so_far += 1
            #     else:
            #         context_input_idxs_filt.append(idx - num_filtered_so_far)
            # context_input_idxs = context_input_idxs_filt
        if len(context_input_idxs) == 0:
            filtered_cluster_indices.append(i)
            continue
        elif len(context_input_idxs) == 1:  # only 1 example
            nns_merged.append(nns[context_input_idxs[0]])
            # already sorted, don't need to sort more
            dists_merged.append(dists[dists_idx])
            entity_mention_bounds_idx.append(np.zeros(dists[dists_idx].shape, dtype=int))
        else:  # merge refering to same example
            all_distances = np.concatenate([dists[dists_idx + j] for j in range(len(context_input_idxs))], axis=-1)
            all_cand_outputs = np.concatenate([nns[context_input_idxs[j]] for j in range(len(context_input_idxs))], axis=-1)
            dist_sort_idx = np.argsort(-all_distances)  # get in descending order
            nns_merged.append(all_cand_outputs[dist_sort_idx])
            dists_merged.append(all_distances[dist_sort_idx])

            # selected_mention_idx
            # [0,len(dists[0])-1], [len(dists[0]),2*len(dists[0])-1], etc. same range all refer to same mention
            # idx of mention bounds corresponding to entity at nns[example][i]
            entity_mention_bounds_idx.append((dist_sort_idx / len(dists[0])).astype(int))

        for i in range(len(context_input_idxs)):
            assert labels[context_input_idxs[0]] == labels[context_input_idxs[i]]
            assert samples[context_input_idxs[0]]["q_id"] == samples[context_input_idxs[i]]["q_id"]
            assert samples[context_input_idxs[0]]["label"] == samples[context_input_idxs[i]]["label"]
            assert samples[context_input_idxs[0]]["label_id"] == samples[context_input_idxs[i]]["label_id"]
            if "gold_context_left" in samples[context_input_idxs[0]]:
                assert samples[context_input_idxs[0]]["gold_context_left"] == samples[context_input_idxs[i]]["gold_context_left"]
                assert samples[context_input_idxs[0]]["gold_mention"] == samples[context_input_idxs[i]]["gold_mention"]
                assert samples[context_input_idxs[0]]["gold_context_right"] == samples[context_input_idxs[i]]["gold_context_right"]
            if "all_gold_entities" in samples[context_input_idxs[0]]:
                assert samples[context_input_idxs[0]]["all_gold_entities"] == samples[context_input_idxs[i]]["all_gold_entities"]
        
        labels_merged.append(labels[context_input_idxs[0]])
        samples_merged.append({
            "q_id": samples[context_input_idxs[0]]["q_id"],
            "label": samples[context_input_idxs[0]]["label"],
            "label_id": samples[context_input_idxs[0]]["label_id"],
            "context_left": [samples[context_input_idxs[j]]["context_left"] for j in range(len(context_input_idxs))],
            "mention": [samples[context_input_idxs[j]]["mention"] for j in range(len(context_input_idxs))],
            "context_right": [samples[context_input_idxs[j]]["context_right"] for j in range(len(context_input_idxs))],
        })
        if "gold_context_left" in samples[context_input_idxs[0]]:
            samples_merged[len(samples_merged)-1]["gold_context_left"] = samples[context_input_idxs[0]]["gold_context_left"]
            samples_merged[len(samples_merged)-1]["gold_mention"] = samples[context_input_idxs[0]]["gold_mention"]
            samples_merged[len(samples_merged)-1]["gold_context_right"] = samples[context_input_idxs[0]]["gold_context_right"]
        if "all_gold_entities" in samples[context_input_idxs[0]]:
            samples_merged[len(samples_merged)-1]["all_gold_entities"] = samples[context_input_idxs[0]]["all_gold_entities"]
        dists_idx += len(context_input_idxs)
    return samples_merged, labels_merged, nns_merged, dists_merged, entity_mention_bounds_idx, filtered_cluster_indices


def _retrieve_from_saved_biencoder_outs(save_preds_dir):
    labels = np.load(os.path.join(args.save_preds_dir, "biencoder_labels.npy"))
    nns = np.load(os.path.join(args.save_preds_dir, "biencoder_nns.npy"))
    dists = np.load(os.path.join(args.save_preds_dir, "biencoder_dists.npy"))
    return labels, nns, dists


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
    biencoder_params["mention_aggregation_type"] = args.mention_aggregation_type
    biencoder = load_biencoder(biencoder_params)
    if not args.use_cuda and type(biencoder.model).__name__ == 'DataParallel':
        biencoder.model = biencoder.model.module
    elif args.use_cuda and type(biencoder.model).__name__ != 'DataParallel':
        biencoder.model = torch.nn.DataParallel(biencoder.model)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["eval_batch_size"] = args.eval_batch_size
            crossencoder_params["path_to_model"] = args.crossencoder_model
            crossencoder_params["no_cuda"] = not args.use_cuda
        crossencoder = load_crossencoder(crossencoder_params)
        if not args.use_cuda and type(crossencoder.model).__name__ == 'DataParallel':
            crossencoder.model = crossencoder.model.module
        elif args.use_cuda and type(biencoder.model).__name__ != 'DataParallel':
            crossencoder.model = torch.nn.DataParallel(crossencoder.model)

    # load candidate entities
    logger.info("loading candidate entities")

    if args.debug_biencoder:
        candidate_encoding, candidate_token_ids, title2id, id2title, id2text, kb2id, id2kb = _load_candidates(
            "/private/home/belindali/temp/BLINK-Internal/models/entity_debug.jsonl",
            "/private/home/belindali/temp/BLINK-Internal/models/entity_encode_debug.t7",
            "/private/home/belindali/temp/BLINK-Internal/models/entity_ids_debug.t7",  # TODO MAKE THIS FILE!!!!
            biencoder, biencoder_params["max_cand_length"], args.entity_catalogue == args.test_entities, logger=logger
        )
    else:
        (
            candidate_encoding,
            candidate_token_ids,
            title2id,
            id2title,
            id2text,
            wikipedia_id2local_id,
            kb2id,
            id2kb,
        ) = _load_candidates(
            args.entity_catalogue, args.entity_encoding, args.entity_token_ids,
            biencoder, biencoder_params["max_cand_length"], args.entity_catalogue == args.test_entities,
            logger=logger
        )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        candidate_token_ids,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        kb2id,
        id2kb,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    candidate_token_ids,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    kb2id,
    id2kb,
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

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = _annotate(ner_model, [text])

            _print_colorful_text(text, samples)
        elif args.qa_data:
            logger.info("QA (WebQSP/GraphQs/NQ) dataset mode")
            # EL for QAdata mode

            #if args.do_ner == "flair":
            #    # Load NER model
            #    ner_model = NER.get_model()

            #    lines = json.load(open(args.test_mentions))
            #    text = [line['utterance'] for line in lines]

            #    # Identify mentions
            #    samples = _annotate(ner_model, text)

            #    kb2id, id2kb = __map_test_entities(args.test_entities, title2id, logger)
            #else:
            logger.info("Loading test samples....")
            samples, kb2id, id2kb, num_unk, sample_to_all_context_inputs = _get_test_samples(
                args.test_mentions, args.test_entities, title2id, kb2id, id2kb, logger,
                qa_data=True, do_ner=args.do_ner, debug=args.debug_biencoder,
                main_entity_only=args.eval_main_entity, do_map_test_entities=(len(kb2id) == 0)
            )
            logger.info("Finished loading test samples")

            if args.debug_biencoder:
                sample_to_all_context_inputs = sample_to_all_context_inputs[:10]
                samples = samples[:sample_to_all_context_inputs[-1][-1] + 1]

            stopping_condition = True
        else:
            logger.info("test dataset mode")

            # Load test mentions
            samples, _, _, _, _ = _get_test_samples(
                args.test_mentions, args.test_entities, title2id, kb2id, id2kb, logger
            )
            stopping_condition = True

        # prepare the data for biencoder
        # run biencoder if predictions not saved
        if not args.qa_data or not os.path.exists(os.path.join(args.save_preds_dir, 'biencoder_labels.npy')):
            logger.info("Preparing data for biencoder....")
            dataloader = _process_biencoder_dataloader(
                samples, biencoder.tokenizer, biencoder_params
            )
            logger.info("Finished preparing data for biencoder")

            # run biencoder
            logger.info("Running biencoder...")
            top_k = 100

            labels, nns, dists = _run_biencoder(biencoder, dataloader, candidate_encoding, top_k,
                device="cpu" if biencoder_params["no_cuda"] else "cuda")
            logger.info("Finished running biencoder")
            
            np.save(os.path.join(args.save_preds_dir, "biencoder_labels.npy"), labels)
            np.save(os.path.join(args.save_preds_dir, "biencoder_nns.npy"), nns)
            np.save(os.path.join(args.save_preds_dir, "biencoder_dists.npy"), dists)
        else:
            labels, nns, dists = _retrieve_from_saved_biencoder_outs(args.save_preds_dir)

        logger.info("Merging inputs...")
        samples_merged, labels_merged, nns_merged, dists_merged, entity_mention_bounds_idx, _ = _combine_same_inputs_diff_mention_bounds(
            samples, labels, nns, dists, sample_to_all_context_inputs,
        )
        logger.info("Finished merging inputs")

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                _print_colorful_prediction(idx, sample, e_id, e_title, e_text)
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        elif args.qa_data:
            # save biencoder predictions and print precision/recalls
            do_sort = False
            entity_freq_map = {}
            entity_freq_map[""] = 0  # for null cases
            with open("/private/home/belindali/starsem2018-entity-linking/resources/wikidata_entity_freqs.map") as f:
                for line in f:
                    split_line = line.split("\t")
                    entity_freq_map[split_line[0]] = int(split_line[1])
            num_correct = 0
            num_predicted = 0
            num_gold = 0
            save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
            all_entity_preds = []
            # save predictions
            with open(save_biencoder_file, 'w') as f:
                for i, entity_list in enumerate(nns_merged):
                    if do_sort:
                        entity_list = entity_list.tolist()
                        entity_list.sort(key=(lambda x: entity_freq_map.get(id2kb.get(x, ""), 0)), reverse=True)
                    e_id = entity_list[0]
                    e_kbid = id2kb.get(e_id, "")
                    pred_kbids_sorted = []
                    for all_id in entity_list:
                        kbid = id2kb.get(all_id, "")
                        pred_kbids_sorted.append(kbid)
                    e_mention_bounds = int(entity_mention_bounds_idx[i][0])
                    label = labels_merged[i]
                    sample = samples_merged[i]
                    distances = dists_merged[i]
                    input = ["{}[{}]{}".format(
                        sample['context_left'][i],
                        sample['mention'][i], 
                        sample['context_right'][i],
                    ) for i in range(len(sample['context_left']))]
                    if 'gold_context_left' in sample:
                        gold_mention_bounds = "{}[{}]{}".format(sample['gold_context_left'],
                            sample['gold_mention'], sample['gold_context_right'])
                    # assert input == first_input
                    # assert label == first_label
                    # f.write(e_kbid + "\t" + str(sample['label']) + "\t" + str(input) + "\n")

                    if args.eval_main_entity:
                        if e_kbid != "":
                            gold_triple = [(
                                sample['label'],
                                len(sample['gold_context_left']), 
                                len(sample['gold_context_left']) + len(sample['gold_mention']),
                            )]
                            pred_triple = [(
                                e_kbid,
                                len(sample['context_left'][e_mention_bounds]), 
                                len(sample['context_left'][e_mention_bounds]) + len(sample['mention'][e_mention_bounds]),
                            )]
                            if entity_linking_tp_with_overlap(gold_triple, pred_triple):
                                num_correct += 1
                        num_total += 1
                    else:
                        if "all_gold_entities" in sample:
                            # get *all* entities, removing duplicates
                            all_pred_entities = set(pred_kbids_sorted[:1])
                            for entity in all_pred_entities:
                                if entity in sample["all_gold_entities"]:
                                    num_correct += 1
                                num_predicted += 1
                            num_gold += len(sample["all_gold_entities"])
                        
                    # if sample['label'] is not None:
                    #     num_predicted += 1
                    entity_results = {
                        "q_id": sample["q_id"],
                        "top_KBid": e_kbid,
                        "all_gold_entities": sample.get("all_gold_entities", None),
                        # "id": e_id,
                        # "title": e_title,
                        # "text": e_text,
                        "all_pred_KBids": [id2kb.get(e_id, "") for e_id in entity_list],
                        "input_mention_bounds": input,
                        "top_mention_bound_idx": e_mention_bounds,
                        "gold_KBid": sample['label'],
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
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                else:
                    f1 = 0
                print("biencoder precision = {} / {} = {}".format(num_correct, num_predicted, p))
                print("biencoder recall = {} / {} = {}".format(num_correct, num_gold, r))
                print("biencoder f1 = {}".format(f1))
            if args.do_ner == "none":
                print("number unknown entity examples: {}".format(num_unk))

            # get recall values
            top_k = 100
            x = []
            y = []
            for i in range(1, top_k):
                temp_y = 0.0
                for label, top in zip(labels_merged, nns_merged):
                    if label in top[:i]:
                        temp_y += 1
                if len(labels_merged) > 0:
                    temp_y /= len(labels_merged)
                x.append(i)
                y.append(temp_y)
            # plt.plot(x, y)
            biencoder_accuracy = y[0]
            recall_at = y[-1]
            print("biencoder accuracy: %.4f" % biencoder_accuracy)
            print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

            if args.fast:
                # use only biencoder
                return biencoder_accuracy, recall_at, 0, 0, len(samples)

        # prepare crossencoder data
        logger.info("preparing crossencoder data")
        # if args.debug_cross:
        #     samples = samples[:10]
        #     labels = labels[:10]
        #     nns = nns[:10]

        # prepare crossencoder data
        if not os.path.exists(os.path.join(args.save_preds_dir, "context_input.t7")):
            context_input, candidate_input, label_input, filtered_indices = prepare_crossencoder_data(
                crossencoder.tokenizer, candidate_token_ids,
                samples, labels, nns,
                id2title, id2text,
                keep_all=(args.get_predictions or args.interactive),
                logger=logger,
            )
            torch.save(context_input, os.path.join(args.save_preds_dir, "context_input.t7"))
            torch.save(candidate_input, os.path.join(args.save_preds_dir, "candidate_input.t7"))
            torch.save(label_input, os.path.join(args.save_preds_dir, "label_input.t7"))
            torch.save(filtered_indices, os.path.join(args.save_preds_dir, "filtered_indices.t7"))
        else:
            context_input = torch.load(os.path.join(args.save_preds_dir, "context_input.t7"))
            candidate_input = torch.load(os.path.join(args.save_preds_dir, "candidate_input.t7"))
            label_input = torch.load(os.path.join(args.save_preds_dir, "label_input.t7"))
            filtered_indices = torch.load(os.path.join(args.save_preds_dir, "filtered_indices.t7"))

        save_crossencoder_file = os.path.join(args.save_preds_dir, 'crossencoder_outs.npy')

        # run crossencoder and get accuracy
        if not os.path.exists(save_crossencoder_file):
            logger.info("Merging context+candidate input for crossencoder...")
            context_input = modify(
                context_input, candidate_input, crossencoder_params["max_seq_length"]
            )

            logger.info("Creating dataloader for crossencoder...")
            dataloader = _process_crossencoder_dataloader(
                context_input, label_input, crossencoder_params
            )
            logger.info("Running crossencoder...")
            if args.debug_cross:
                accuracy, res = _run_crossencoder(
                    crossencoder, dataloader, logger, device=("cuda" if args.use_cuda else "cpu"), context_len=biencoder_params["max_context_length"],
                    id2kb=id2kb, get_all_scores=True, forward_only=args.get_predictions,
                )
                # save scores
            else:
                accuracy, res = _run_crossencoder(
                    crossencoder, dataloader, logger, device=("cuda" if args.use_cuda else "cpu"), context_len=biencoder_params["max_context_length"],
                    get_all_scores=True, forward_only=args.get_predictions
                )

            logger.info("Finished running crossencoder")
            np.save(save_crossencoder_file, res["logits"])
            print("crossencoder accuracy (unmerged same examples, different entities): %d / %d = %.4f" % (res['num_correct'], res['num_total'], accuracy))
        else:
            res = {
                "logits": np.load(save_crossencoder_file)
            }


        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            _print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, prediction, sample in zip(nns, predictions, samples):
                e_id = entity_list[prediction]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                _print_colorful_prediction(idx, sample, e_id, e_title, e_text)
                idx += 1
            print()

        elif args.qa_data:
            filtered_indices = set(filtered_indices)
            
            if args.debug_cross:
                import pdb
                pdb.set_trace()
            samples_merged, labels_merged, nns_merged, scores_merged, entity_mention_bounds_idx, filtered_cluster_indices = \
                _combine_same_inputs_diff_mention_bounds(
                    samples, labels, nns, res["logits"], sample_to_all_context_inputs, debug=args.debug_cross, filtered_indices=filtered_indices
                )
            # filter out missing clusters
            # all_entity_preds = [all_entity_preds[i] for i in range(len(all_entity_preds)) if i not in filtered_cluster_indices]
            if not args.debug_cross:
                assert len(samples_merged) == len(all_entity_preds)
            
            # save crossencoder predictions
            save_crossencoder_file_json = os.path.join(args.save_preds_dir, 'crossencoder_outs.jsonl')
            with open(save_crossencoder_file_json, "w") as f:
                num_correct = 0
                num_predicted = 0
                num_gold = 0
                num_total = 0
                cross_results_idx = 0  # indexes into cross encoder results i.e. res[xx]
                for i, example in enumerate(all_entity_preds):
                    # don't have estimates
                    # TODO do this in biencoder eval as well
                    sample = samples_merged[i]
                    example_scores = scores_merged[i]
                    pred_kbids_sorted = []
                    pred_titles_sorted = []
                    for el_id in nns_merged[i]:
                        kbid = id2kb.get(el_id, "")
                        title = id2title.get(el_id, "")
                        pred_kbids_sorted.append(kbid)
                        pred_titles_sorted.append(title)
                        # sanity checks
                        if kbid != "":
                            assert title != ""
                        assert kbid in example["all_pred_KBids"]

                    example_cross_results = {
                        "q_id": example["q_id"],
                        "input_mention_bounds": example["input_mention_bounds"],
                        "sorted_pred_KBids": pred_kbids_sorted,
                        "sorted_pred_titles": pred_titles_sorted,
                        "sorted_mention_bound_idx": entity_mention_bounds_idx[i].tolist(),
                        "scores": example_scores.tolist(),
                    }
                    if "gold_KBid" in example or "all_gold_entities" in sample:
                        if args.eval_main_entity:
                            if len(pred_kbids_sorted) == 0:
                                # preds were all skipped (filtered out)
                                example_cross_results["top_pred_KBid"] = "not_in_bienc_cands"
                                pred_triple = [(
                                    example_cross_results["top_pred_KBid"], 0, 0,
                                )]
                            else:
                                example_cross_results["top_pred_KBid"] = pred_kbids_sorted[0]
                                pred_triple = [(
                                    example_cross_results["top_pred_KBid"],
                                    len(sample['context_left'][entity_mention_bounds_idx[i][0]]), 
                                    len(sample['context_left'][entity_mention_bounds_idx[i][0]]) + len(sample['mention'][entity_mention_bounds_idx[i][0]]),
                                )]
                            gold_triple = [(
                                example["gold_KBid"],
                                len(sample['gold_context_left']), 
                                len(sample['gold_context_left']) + len(sample['gold_mention']),
                            )]
                            if entity_linking_tp_with_overlap(gold_triple, pred_triple):
                                num_correct += 1
                            num_total += 1
                            example_cross_results["gold_KBid"] = example["gold_KBid"]
                        else:
                            # get *all* entities, removing duplicates
                            all_pred_entities = set(pred_kbids_sorted[:1])
                            for entity in all_pred_entities:
                                if entity in sample["all_gold_entities"]:
                                    num_correct += 1
                                num_predicted += 1
                            num_gold += len(sample["all_gold_entities"])
                            example_cross_results["all_gold_entities"] = sample["all_gold_entities"]

                    f.write(json.dumps(example_cross_results) + "\n")
            if args.eval_main_entity:
                if num_total > 0:
                    print("crossencoder accuracy (merged same examples, different entities): %d / %d = %.4f" % (
                        num_correct, num_total, float(num_correct) / num_total,
                    ))
            else:
                if num_gold and num_predicted > 0:
                    print("crossencoder precision (merged same examples, different entities): %d / %d = %.4f" % (
                        num_correct, num_predicted, float(num_correct) / num_predicted,
                    ))
                    print("crossencoder recall (merged same examples, different entities): %d / %d = %.4f" % (
                        num_correct, num_gold, float(num_correct) / num_gold,
                    ))
                    print("crossencoder F1 (merged same examples, different entities): %.4f" % (
                        (2 * (float(num_correct) / num_gold) * (float(num_correct) / num_predicted)) / (float(num_correct) / num_gold + float(num_correct) / num_predicted),
                    ))

            print("Finished saving crossencoder outputs")
        else:
            crossencoder_normalized_accuracy = accuracy
            print(
                "crossencoder normalized accuracy: %.4f"
                % crossencoder_normalized_accuracy
            )

            overall_unormalized_accuracy = (
                crossencoder_normalized_accuracy * len(label_input) / len(samples)
            )
            print("overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy)
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                len(samples),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug", "-d", action="store_true", default=False, help="Run debug mode"
    )
    parser.add_argument(
        "--debug_biencoder", "-db", action="store_true", default=False, help="Debug biencoder"
    )
    parser.add_argument(
        "--debug_cross", "-dc", action="store_true", default=False, help="Debug crossencoder"
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
        "--do_ner", "-n", type=str, default='none', choices=['flair', 'ngram', 'single', 'qa_classifier', 'none'],
        help="Use automatic NER systems. Options: 'flair', 'ngram', 'single', 'qa_classifier' 'none'."
        "(Set 'none' to get gold mention bounds from examples)"
    )
    parser.add_argument(
        "--qa_classifier_threshold", type=str, default=None, help="Must be specified if '--do_ner qa_classifier'."
        "Threshold for qa classifier score for which examples will be pruned if they fall under that threshold."
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
    # TODO DELETE LATER
    parser.add_argument(
        "--no_mention_bounds_biencoder",
        dest="no_mention_bounds_biencoder",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mention_aggregation_type",
        default=None,
        type=str,
        help="Type of mention aggregation (None to just use [CLS] token, "
        "'all_avg' to average across tokens in mention, 'fl_avg' to average across first/last tokens in mention, "
        "'{all/fl}_linear' for linear layer over mention, '{all/fl}_mlp' to MLP over mention)",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
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
