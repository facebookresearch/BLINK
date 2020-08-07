import json
import os
import numpy as np
import torch
from blink.vcg_utils.measures import entity_linking_tp_with_overlap
from tqdm import tqdm

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


id2title = json.load(open("models/id2title.json"))

def load_dists(all_save_dir, data, split, model, joint_threshold):
    save_dir = "{}/{}_{}_{}_joint{}_top50cands_final_joint".format(all_save_dir, data, split, model, joint_threshold)
    if not os.path.exists(save_dir):
        save_dir += "_0"
    with open(os.path.join(save_dir, "biencoder_outs.jsonl")) as f:
        examples = f.readlines()
        examples = [json.loads(line) for line in examples]
    biencoder_indices = np.load(os.path.join(save_dir, "biencoder_nns.npy"), allow_pickle=True)  # corresponds to biencoder_dists
    biencoder_dists = np.load(os.path.join(save_dir, "biencoder_dists.npy"), allow_pickle=True)
    if os.path.exists(os.path.join(save_dir, "biencoder_cand_scores.npy")):
        cand_dists = np.load(os.path.join(save_dir, "biencoder_cand_scores.npy"), allow_pickle=True)
    else:
        cand_dists = np.load(os.path.join(save_dir, "biencoder_cand_dists.npy"), allow_pickle=True)
    pred_mention_bounds = np.load(os.path.join(save_dir, "biencoder_mention_bounds.npy"), allow_pickle=True)
    if os.path.exists(os.path.join(save_dir, "biencoder_mention_scores.npy")):
        mention_dists = np.load(os.path.join(save_dir, "biencoder_mention_scores.npy"), allow_pickle=True)
    else:
        mention_dists = [biencoder_dists[i] - torch.log_softmax(torch.tensor(cand_dists[i]), 1).numpy() for i in range(len(biencoder_dists))]
        # inverse sigmoid
        mention_dists = [np.log(md / (1 - md)) for md in mention_dists]
    return examples, biencoder_indices, biencoder_dists, cand_dists, pred_mention_bounds, mention_dists


def filter_repeats(pred_triples, pred_scores):
    # sort pred_triples and pred_scores by pred_scores
    score_sort_ids = sorted(enumerate(pred_scores), key=lambda x: x[1], reverse=True)
    pred_triples = [pred_triples[si[0]] for si in score_sort_ids]
    pred_scores = [score_sort_id[1] for score_sort_id in score_sort_ids]

    all_pred_entities = {}
    all_pred_entities_pruned = []
    all_pred_scores_pruned = []
    for idx, ent in enumerate(pred_triples):
        if ent[0] in all_pred_entities:
            continue
        all_pred_entities_pruned.append(ent)
        all_pred_scores_pruned.append(pred_scores[idx])
        all_pred_entities[ent[0]] = 0
    return all_pred_entities_pruned, all_pred_scores_pruned


def filter_overlaps(tokens, pred_triples, pred_scores):
    all_pred_entities_pruned = []
    all_pred_scores_pruned = []
    mention_masked_utterance = np.zeros(len(tokens))
    # mention_masked_utterance = np.zeros(len(tokens))
    # ensure well-formed-ness, prune overlaps
    # greedily pick highest scoring, then prune all overlapping
    for idx, mb in enumerate(pred_triples):
        if sum(mention_masked_utterance[mb[1]:mb[2]]) > 0:
            continue
        all_pred_entities_pruned.append(mb)
        all_pred_scores_pruned.append(pred_scores[idx])
        mention_masked_utterance[mb[1]:mb[2]] = 1
    return all_pred_entities_pruned, all_pred_scores_pruned


def filter_repeat_overlaps(tokens, pred_triples, pred_scores):
    all_pred_entities_pruned = []
    all_pred_scores_pruned = []
    mention_masked_utterance = {triple[0]: np.zeros(len(tokens)) for triple in pred_triples}
    # mention_masked_utterance = np.zeros(len(tokens))
    # ensure well-formed-ness, prune overlaps
    # greedily pick highest scoring, then prune all overlapping
    for idx, mb in enumerate(pred_triples):
        if sum(mention_masked_utterance[mb[0]][mb[1]:mb[2]]) > 0:
            continue
        all_pred_entities_pruned.append(mb)
        all_pred_scores_pruned.append(pred_scores[idx])
        mention_masked_utterance[mb[0]][mb[1]:mb[2]] = 1
    return all_pred_entities_pruned, all_pred_scores_pruned


# threshold and sort by score
def get_threshold_mask_and_sort(mention_dists, cand_dists, biencoder_dists, valid_cands_mask, threshold, top_mention_sort=True):
    """
        top_mention_sort:
            True: sort top candidates per mention only
                scores_mask and sorted_idxs has dim (#_valid_examples,)
            False: sort ALL candidates (assumes multiple candidates per mention)
                scores_mask and sorted_idxs has dim (#_valid_examples, #_cands)
    """
    # (mention_dists[i][:,0] != -1) & (mention_dists[i][:,0] == mention_dists[i][:,0])
    mention_scores = mention_dists[valid_cands_mask]
    if len(mention_scores.shape) > 1:
        mention_scores = mention_scores[:,0]
    # scores = 1/(1 + np.exp(-mention_dists[valid_cands_mask])) + torch.log_softmax(torch.tensor(cand_dists[valid_cands_mask]), 1).numpy()
    # take only the top cands
    # top_pred_entity = pred_entity_list[:,0]
    # top_entity_mention_bounds_idx = entity_mention_bounds_idx[:,0]
    scores = torch.log_softmax(torch.tensor(cand_dists[valid_cands_mask]), 1) + torch.sigmoid(torch.tensor(mention_scores)).log().unsqueeze(-1)  # GRAPHQUESTIONS BEST
    if top_mention_sort:
        # scores_mask = (mention_scores > -3) | (torch.log_softmax(torch.tensor(cand_dists[valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -2).numpy()
        scores_mask = (scores[:,0] > threshold)  # GRAPHQUESTIONS BEST
        # scores_mask = (torch.sigmoid(torch.tensor(mention_scores)) > 0.25).numpy() & (biencoder_dists[valid_cands_mask][:,0] > threshold)  # WEBQSP BEST
        # scores_mask = (mention_scores > -3) & (torch.log_softmax(torch.tensor(cand_dists[valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -5).numpy()
        # scores_mask = (mention_scores > -3) & (torch.log_softmax(torch.tensor(cand_dists[valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -2).numpy()
        # scores_mask = (mention_scores > -5) & ((mention_scores + cand_dists[valid_cands_mask][:,0]) > 5)
        # scores_mask = (torch.sigmoid(torch.tensor(mention_scores)) > 0.1).numpy() & (torch.log_softmax(torch.tensor(biencoder_dists[valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)) > 0).numpy()
        # scores_mask = (mention_scores > -float("inf"))
        # sort...
        # _, sorted_idxs = (torch.log_softmax(torch.tensor(biencoder_dists[valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log())[scores_mask].sort()
        # _, sorted_idxs = (torch.log_softmax(torch.tensor(biencoder_dists[i][valid_cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log())[scores_mask].sort()
        _, sorted_idxs = scores[:,0][scores_mask].sort(descending=True)
        sorted_filtered_scores = scores[scores_mask][sorted_idxs]
        # _, sorted_idxs = torch.tensor(biencoder_dists[valid_cands_mask][:,0][scores_mask]).sort(descending=True)  # WEBQSP BEST
        # _, sorted_idxs = torch.tensor(mention_scores[scores_mask]).sort()
    else:
        scores_mask = (scores > threshold)  # GRAPHQUESTIONS BEST
        try:
            sorted_filtered_scores, sorted_idxs = scores[scores_mask].sort(descending=True)
        except:
            import pdb
            pdb.set_trace()
    return scores_mask.numpy(), sorted_idxs.numpy(), sorted_filtered_scores.numpy()


all_save_dir = "/checkpoint/belindali/entity_link/saved_preds"
model_type = "finetuned_webqsp"  # wiki
if model_type == "wiki":
    model = '{0}_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;15'.format(model_type)
elif model_type == "finetuned_webqsp":
    model= '{0}_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;18'.format(model_type)

get_topk_cands = False
topk = 100
if get_topk_cands:
    threshold=-float("inf")
else:
    # threshold=0
    # threshold=-2.9
    # threshold=-1.5
    threshold=-5

for data in ["nq", "WebQuestions", "triviaqa"]:
    if data == "nq":
        splits = ["dev", "test", "train0", "train1", "train2"]
    else:
        splits = ["train", "dev", "test"]
    for split in splits:

f1s = []
for threshold in [-0.8, -0.9, -1, -1.1, -1.2]:
(
    examples, biencoder_indices, biencoder_dists,
    cand_dists, pred_mention_bounds, mention_dists
) = load_dists(all_save_dir, data, split, model, "0.0" if model_type == "wiki" else "-inf")
new_examples = []
num_correct=0
num_predicted=0
num_gold=0
for i, example in enumerate(tqdm(examples)):
    # select valid candidates
    valid_cands_mask = (biencoder_dists[i][:,0] != -1) & (biencoder_dists[i][:,0] == biencoder_dists[i][:,0])
    # get scores and masking/sorting by score
    scores_mask, sorted_idxs, sorted_filtered_scores = get_threshold_mask_and_sort(
        mention_dists[i], cand_dists[i], biencoder_dists[i], valid_cands_mask, threshold, top_mention_sort=(not get_topk_cands)
    )
    if get_topk_cands:
        # (filtered_examples, #cands, 2)
        ex_pred_mention_bounds = np.repeat(np.expand_dims(pred_mention_bounds[i], axis=1), biencoder_indices[i].shape[1], axis=1)
        # (filtered_examples, #cands,)
        ex_mention_dists = np.repeat(np.expand_dims(mention_dists[i], axis=1), biencoder_indices[i].shape[1], axis=1)
        ex_biencoder_indices = biencoder_indices[i]
        ex_cand_dists = cand_dists[i]
    else:
        ex_pred_mention_bounds = pred_mention_bounds[i]
        ex_mention_dists = mention_dists[i]
        ex_biencoder_indices = biencoder_indices[i]  #[:,0]
        ex_cand_dists = cand_dists[i]  #[:,0]
    # output threshold_entities_translate, pred_triples, pred_scores
    threshold_entities = ex_biencoder_indices[valid_cands_mask][scores_mask][sorted_idxs]  # (filtered_exs, #cands) / (filtered_cands,)
    threshold_mention_bounds = ex_pred_mention_bounds[valid_cands_mask][scores_mask][sorted_idxs]  # (filtered_exs, 2) / (filtered_cands, 2)
    threshold_cand_scores = ex_cand_dists[valid_cands_mask][scores_mask][sorted_idxs]  # (filtered_exs, #cands) / (filtered_cands,)
    threshold_mention_scores = ex_mention_dists[valid_cands_mask][scores_mask][sorted_idxs]  # (filtered_exs,) / (filtered_cands,)
    threshold_scores = sorted_filtered_scores  # (filtered_exs, #cands) / (filtered_cands,)
    threshold_entities_translate = {}
    pred_triples = []
    pred_scores = []
    example['tokens'] = [101] + example['tokens'] + [102]
    for m in range(len(threshold_scores)):
        mb = threshold_mention_bounds[m].tolist()
        mention_text = tokenizer.decode(example['tokens'][mb[0]:mb[1]+1])
        threshold_entities_translate[mention_text] = {
            "mention_idx": m, "mention_score": float(threshold_mention_scores[m])
        }
        if len(threshold_entities[m].shape) > 0:
            pred_triples.append([str(threshold_entities[m][0]), mb[0], mb[1]+1])
            pred_scores.append(float(threshold_scores[m][0]))
            threshold_entities_translate[mention_text]["candidate_entities"] = []
            threshold_entities_translate[mention_text]["cand_scores"] = threshold_cand_scores[m].tolist()
            for id in threshold_entities[m]:
                threshold_entities_translate[mention_text]["candidate_entities"].append(id2title[str(id)])
        else:
            pred_triples.append([str(threshold_entities[m]), mb[0], mb[1]+1])
            pred_scores.append(float(threshold_scores[m]))
            threshold_entities_translate[mention_text]["candidate_entities"] = id2title[str(threshold_entities[m])]
            threshold_entities_translate[mention_text]["cand_scores"] = float(threshold_cand_scores[m])
    new_ex = {
        "id": example["id"],
        "text": example["text"],
        "tokens": example["tokens"],
    }
    if "gold_triples" in example:
        all_pred_entities_pruned = pred_triples
        all_pred_scores_pruned = pred_scores
        if get_topk_cands:
            all_pred_entities_pruned, all_pred_scores_pruned = filter_repeats(pred_triples, pred_scores)
            all_pred_entities_pruned = all_pred_entities_pruned[:topk]
            all_pred_scores_pruned = all_pred_scores_pruned[:topk]
        else:
            all_pred_entities_pruned, all_pred_scores_pruned = filter_overlaps(example["tokens"], pred_triples, pred_scores)
    else:
        all_pred_entities_pruned = pred_triples
        all_pred_scores_pruned = pred_scores
        if get_topk_cands:
            all_pred_entities_pruned, all_pred_scores_pruned = filter_repeats(pred_triples, pred_scores)
            all_pred_entities_pruned = all_pred_entities_pruned[:topk]
            all_pred_scores_pruned = all_pred_scores_pruned[:topk]
        else:
            all_pred_entities_pruned, all_pred_scores_pruned = filter_overlaps(example["tokens"], pred_triples, pred_scores)
    new_ex['pred_mentions'] = threshold_entities_translate
    new_ex['pred_triples'] = [[triple[0], triple[1]-1, triple[2]-1] for triple in all_pred_entities_pruned]
    new_ex['pred_triples_score'] = all_pred_scores_pruned
    new_ex['pred_triples_string'] = [
        [id2title[triple[0]], tokenizer.decode(example['tokens'][triple[1]:triple[2]])]
        for triple in all_pred_entities_pruned
    ]
    # get scores
    if "gold_triples" in example:
        gold_triples = example["gold_triples"]
        new_ex["gold_triples"] = gold_triples
        num_overlap_weak, num_overlap_strong = entity_linking_tp_with_overlap(gold_triples, new_ex['pred_triples'])
        num_correct += num_overlap_weak
        num_predicted += len(all_pred_entities_pruned)
        num_gold += len(gold_triples)
    new_examples.append(new_ex)

# compute metrics
if num_predicted > 0 and num_gold > 0:
    p = num_correct / num_predicted
    r = num_correct / num_gold
    f1 = 2*p*r / (p+r)
    print(f1)
    f1s.append(f1)

        if get_topk_cands:
            print("Saving {} {} {}".format(data, split, str(topk)))
            save_file = "{}_{}_top{}.jsonl".format(split, model_type, str(topk))
        else:
            print("Saving {} {} {}".format(data, split, str(threshold)))
            save_file = "{}_{}_{}.jsonl".format(split, model_type, str(threshold))
        # save
        with open(os.path.join("/checkpoint/belindali/entity_link/data/{}/saved_preds".format(data), save_file), 'w') as wf:
            for new_ex in new_examples:
                b=wf.write(json.dumps(new_ex) + "\n")