import torch
import numpy as np
import json
import os

from blink.main_dense import _load_candidates
from tqdm import tqdm

from blink.vcg_utils.measures import entity_linking_tp_with_overlap

import torch
import itertools

entity_catalogue = "/private/home/belindali/pretrain/BLINK-mentions/models/entity.jsonl"
kb2id, id2kb = _load_kbids(entity_catalogue)

SAVE_PREDS_DIR = "/checkpoint/belindali/entity_link/saved_preds/webqsp_filtered_dev_finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_bert_large_qa_linear_joint0.3_top10cands_final_joint_0"

SAVE_PREDS_DIR = "/checkpoint/belindali/entity_link/saved_preds/graphqs_filtered_dev_finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_bert_large_qa_linear_joint0.35_top12cands_final_joint_0"
SAVE_PREDS_DIR = "/checkpoint/belindali/entity_link/saved_preds/graphqs_filtered_dev_finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_bert_large_qa_linear_joint0.35_top11cands_final_joint_0"
SAVE_PREDS_DIR = "/checkpoint/belindali/entity_link/saved_preds/webqsp_filtered_dev_finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_bert_large_qa_linear_joint0.2_top6cands_final_joint_0"

SAVE_PREDS_DIR = "/checkpoint/belindali/entity_link/saved_preds/webqsp_filtered_dev_finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_bert_large_qa_linear_joint0.2_top7cands_final_joint_0/"


(dists, cand_dists, mention_dists, labels, nns, runtime, new_samples, sample_to_all_context_inputs, samples_all) = load_files(SAVE_PREDS_DIR)
(dists_top10, cand_dists_top10, mention_dists_top10, _, nns_top10, _, _, _, samples_all_top10) = load_files(SAVE_PREDS_DIR)


cand_dists_topk,  = torch.tensor(cand_dists).topk(5)

# A = [0.2 + 0.01 * i for i in range(10)]
A = [-1 + 0.1 * i for i in range(21)]
B = [-1 + 0.1 * i for i in range(21)]
# AB = [A for i in range(21)]
C = [-1 + 0.1 * i for i in range(21)]

# >>> max_f1
# 0.5934765314240255
# >>> max_configs
# [[0.40000000000000013, 1, 0]]

# >>> max_f1
# 0.6083018867924529
# >>> max_configs
# [[0.40000000000000013, 1.0, 0.9000000000000001]]

# >>> max_configs
# [[0.20000000000000018, 0.20000000000000018, 0.30000000000000004], [0.40000000000000013, 0.40000000000000013, 0.6000000000000001], [0.6000000000000001, 0.6000000000000001, 0.9000000000000001]]
# >>> max_f1
# 0.8924050632911392

cross = [(x, y, z) for x in A for y in B for z in C]
configs_to_f1 = {}
f1_to_configs = {}
max_configs = [0, 0, 0]
max_f1 = 0.0
for abc in tqdm(cross):
    a = abc[0]
    b = abc[1]
    c = abc[2]

C = [i * 0.1 + 2 for i in range(20)]
for c in C:
new_samples_merged, labels_merged, nns_merged, dists_merged, entity_mention_bounds_idx, no_pred_indices = _combine_same_inputs_diff_mention_bounds(
    new_samples, labels, nns,
    # 0.2 * cand_dists[:,:10] + mention_dists[:,:10],
    torch.log_softmax(torch.tensor(cand_dists),1).numpy() + torch.sigmoid(torch.tensor(mention_dists)).numpy(),
    # a * cand_dists + b * mention_dists + c,
    # 0.4 * cand_dists + mention_dists + 0.9,
    sample_to_all_context_inputs,
)
# print(a)
new_samples_merged, labels_merged, nns_merged, dists_merged, entity_mention_bounds_idx, no_pred_indices = _combine_same_inputs_diff_mention_bounds(
    new_samples, labels, nns,
    # 0.2 * cand_dists[:,:10] + mention_dists[:,:10],
    torch.log_softmax(torch.tensor(cand_dists),1).numpy() + 1,#torch.sigmoid(torch.tensor(mention_dists)).numpy(),
    # a * cand_dists + b * mention_dists + c,
    # 0.4 * cand_dists + mention_dists + 0.9,
    sample_to_all_context_inputs,
)
entity_results, f1 = evaluate(samples_all, nns_merged, labels_merged, dists_merged, entity_mention_bounds_idx, runtime, id2kb)
    print(f1)
    configs_to_f1[json.dumps([a, b, c])] = f1
    f1_to_configs[f1] = json.dumps([a, b, c])
    if f1 > max_f1:
        max_f1 = f1
        max_configs = []
    if f1 == max_f1:
        max_configs.append([a, b, c])


def load_files(SAVE_PREDS_DIR):
    dists = np.load(os.path.join(SAVE_PREDS_DIR, "biencoder_dists.npy"))
    cand_dists = np.load(os.path.join(SAVE_PREDS_DIR, "biencoder_cand_dists.npy"))
    mention_dists = dists - cand_dists
    labels = np.load(os.path.join(SAVE_PREDS_DIR, "biencoder_labels.npy"))
    nns = np.load(os.path.join(SAVE_PREDS_DIR, "biencoder_nns.npy"))
    runtime = float(open(os.path.join(SAVE_PREDS_DIR, "runtime.txt")).read())
    new_samples = json.load(open(os.path.join(SAVE_PREDS_DIR, "samples.json")))
    sample_to_all_context_inputs = json.load(open(os.path.join(SAVE_PREDS_DIR, "sample_to_all_context_inputs.json")))
    samples_all = open(os.path.join(SAVE_PREDS_DIR, "biencoder_outs.jsonl")).readlines()
    samples_all = [json.loads(line) for line in samples_all]
    return dists, cand_dists, mention_dists, labels, nns, runtime, new_samples, sample_to_all_context_inputs, samples_all


def _load_kbids(entity_catalogue):
    kb2id = {}
    id2kb = {}
    missing_entity_ids = 0
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for i, line in enumerate(tqdm(lines)):
            entity = json.loads(line)
            if "kb_idx" in entity:
                kb2id[entity["kb_idx"]] = local_idx
                id2kb[local_idx] = entity["kb_idx"]
            else:
                missing_entity_ids += 1
            local_idx += 1
    return kb2id, id2kb


def evaluate(samples_all, nns_merged, labels_merged, dists_merged, entity_mention_bounds_idx, runtime, id2kb):
    num_correct = 0
    num_predicted = 0
    num_gold = 0
    all_entity_preds = []
    for i, sample in enumerate(samples_all):
        entity_list = nns_merged[i]
        pred_kbids_sorted = []
        for all_id in entity_list:
            kbid = id2kb.get(all_id, "")
            pred_kbids_sorted.append(kbid)
        label = labels_merged[i]
        distances = dists_merged[i]
        utterance = sample["gold_mention_bounds"].split(";")[0].replace("[", "").replace("]", "")
        top_indices = np.where(distances > 0)[0]
        # if len(distances) > 0 and len(top_indices) == 0:
        #     top_indices = np.array([0])
        all_pred_entities = [pred_kbids_sorted[topi] for topi in top_indices]
        # already sorted by score
        e_mention_bounds = [entity_mention_bounds_idx[i][topi] for topi in top_indices]
        # prune mention overlaps
        e_mention_bounds_pruned = []
        all_pred_entities_pruned = []
        mention_masked_utterance = np.zeros(len(utterance))
        # ensure well-formed-ness, prune overlaps
        # greedily pick highest scoring, then prune all overlapping
        for idx, mb_idx in enumerate(e_mention_bounds):
            # get mention bounds
            sample['input_mention_bounds'][mb_idx] = sample['input_mention_bounds'][mb_idx].replace("##", "")
            mention_start = sample['input_mention_bounds'][mb_idx].find("[")
            mention_end = sample['input_mention_bounds'][mb_idx].find("]") - 1
            # check if in existing mentions
            try:
                if mention_end >= len(utterance):
                    mention_end = len(utterance)
                if mention_masked_utterance[mention_start] == 1 or mention_masked_utterance[mention_end - 1] == 1:
                    continue
            except:
                import pdb
                pdb.set_trace()
            e_mention_bounds_pruned.append(mb_idx)
            all_pred_entities_pruned.append(all_pred_entities[idx])
            mention_masked_utterance[mention_start:mention_end] = 1
        pred_triples = [(
            all_pred_entities_pruned[j], # TODO REVERT THIS
            sample['input_mention_bounds'][e_mention_bounds_pruned[j]].find("["), 
            sample['input_mention_bounds'][e_mention_bounds_pruned[j]].find("]") - 1,
        ) for j in range(len(all_pred_entities_pruned))]
        gold_triples = sample["gold_triples"]
        num_correct += entity_linking_tp_with_overlap(gold_triples, pred_triples)
        num_predicted += len(all_pred_entities_pruned)
        num_gold += len(sample["all_gold_entities"])
        entity_results = {
            "q_id": sample["q_id"],
            "all_gold_entities": sample.get("all_gold_entities", None),
            "pred_triples": pred_triples,
            "gold_triples": gold_triples,
            "scores": distances.tolist(),
        }
        all_entity_preds.append(entity_results)
    # print()
    p = 0
    r = 0
    if num_predicted > 0:
        p = float(num_correct) / float(num_predicted)
    if num_gold > 0:
        r = float(num_correct) / float(num_gold)
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    # print("biencoder precision = {} / {} = {}".format(num_correct, num_predicted, p))
    # print("biencoder recall = {} / {} = {}".format(num_correct, num_gold, r))
    # print("biencoder f1 = {}".format(f1))
    # print("biencoder runtime = {}".format(runtime))
    return all_entity_preds, f1



def _combine_same_inputs_diff_mention_bounds(samples, labels, nns, dists, sample_to_all_context_inputs, filtered_indices=None, debug=False):
    # TODO save ALL samples
    if not debug:
        try:
            assert len(nns) == sample_to_all_context_inputs[-1][-1] + 1
        except:
            # TODO DEBUG
            import pdb
            pdb.set_trace()
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
        # if len(context_input_idxs) == 0:
        #     # should not happen anymore...
        #     import pdb
        #     pdb.set_trace()
        # first filter all filetered_indices
        if filtered_indices is not None:
            context_input_idxs = [idx for idx in context_input_idxs if idx not in filtered_indices]
        if len(context_input_idxs) == 0:
            filtered_cluster_indices.append(i)
            nns_merged.append(np.array([]))
            dists_merged.append(np.array([]))
            entity_mention_bounds_idx.append(np.array([]))
            labels_merged.append(np.array([]))
            # BOOKMARK
            samples_merged.append({})
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
        if "all_gold_entities_pos" in samples[context_input_idxs[0]]:
            samples_merged[len(samples_merged)-1]["all_gold_entities_pos"] = samples[context_input_idxs[0]]["all_gold_entities_pos"]
        dists_idx += len(context_input_idxs)
    return samples_merged, labels_merged, nns_merged, dists_merged, entity_mention_bounds_idx, filtered_cluster_indices
