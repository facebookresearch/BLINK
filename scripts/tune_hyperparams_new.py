import json
import os
import numpy as np
import torch
from blink.vcg_utils.measures import entity_linking_tp_with_overlap
from tqdm import tqdm

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


id2title = json.load(open("models/id2title.json"))

all_save_dir = "/checkpoint/belindali/entity_link/saved_preds"
data = "nq"
split= "dev"
model= 'wiki_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;15'
# model= 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear'

save_dir = "{}/{}_{}_{}_joint0.0_top50cands_final_joint_0/".format(all_save_dir, data, split, model)

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

# # just change threshold, no changing combination of mention + cand
# def change_threshold(new_threshold):

if os.path.exists(os.path.join(save_dir, "biencoder_mention_scores.npy")):
    mention_dists = np.load(os.path.join(save_dir, "biencoder_mention_scores.npy"), allow_pickle=True)
else:
    mention_dists = [biencoder_dists[i] - torch.log_softmax(torch.tensor(cand_dists[i]), 1).numpy() for i in range(len(biencoder_dists))]
    # inverse sigmoid
    mention_dists = [np.log(md / (1 - md)) for md in mention_dists]


# threshold=-1.9
threshold=-5
new_examples = []
num_correct=0
num_predicted=0
num_gold=0
for i, example in enumerate(tqdm(examples)):
    cands_mask = (biencoder_dists[i][:,0] != -1) & (biencoder_dists[i][:,0] == biencoder_dists[i][:,0])
    # (mention_dists[i][:,0] != -1) & (mention_dists[i][:,0] == mention_dists[i][:,0])
    # (num_pred_mentions, cands_per_mention)
    pred_entity_list = biencoder_indices[i][cands_mask]
    # (num_pred_mentions, 2)
    entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]
    mention_scores = mention_dists[i][cands_mask]
    if len(mention_scores.shape) > 1:
        mention_scores = mention_scores[:,0]
    # scores = 1/(1 + np.exp(-mention_dists[i][cands_mask])) + torch.log_softmax(torch.tensor(cand_dists[i][cands_mask]), 1).numpy()
    # take only the top cands
    # top_pred_entity = pred_entity_list[:,0]
    # top_entity_mention_bounds_idx = entity_mention_bounds_idx[:,0]
    
    # scores_mask = (mention_scores > -3) | (torch.log_softmax(torch.tensor(biencoder_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -2).numpy()
    scores_mask = (torch.log_softmax(torch.tensor(cand_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > threshold).numpy()  # GRAPHQUESTIONS BEST
    # scores_mask = (mention_scores > -3) & (torch.log_softmax(torch.tensor(cand_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -5).numpy()
    # scores_mask = (mention_scores > -3) & (torch.log_softmax(torch.tensor(cand_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log() > -2).numpy()
    # scores_mask = (mention_scores > -5) & ((mention_scores + cand_dists[i][cands_mask][:,0]) > 5)
    # scores_mask = (torch.sigmoid(torch.tensor(mention_scores)) > 0.1).numpy() & (torch.log_softmax(torch.tensor(biencoder_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)) > 0).numpy()
    # scores_mask = (mention_scores > -float("inf"))
    # sort...
    # _, sorted_idxs = (torch.log_softmax(torch.tensor(biencoder_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log())[scores_mask].sort()
    # _, sorted_idxs = (torch.log_softmax(torch.tensor(biencoder_dists[i][cands_mask]), 1)[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log())[scores_mask].sort()
    _, sorted_idxs = ((torch.tensor(cand_dists[i][cands_mask]))[:,0] + torch.sigmoid(torch.tensor(mention_scores)).log())[scores_mask].sort(descending=True)
    # _, sorted_idxs = torch.tensor(mention_scores[scores_mask]).sort()
    threshold_entities = pred_entity_list[scores_mask][sorted_idxs]
    threshold_mention_bounds = entity_mention_bounds_idx[scores_mask][sorted_idxs]
    threshold_scores = cand_dists[i][cands_mask][scores_mask][sorted_idxs]
    threshold_mention_scores = mention_scores[scores_mask][sorted_idxs]
    if len(sorted_idxs) == 1:
        threshold_entities = np.expand_dims(threshold_entities, axis=0)
        threshold_mention_bounds = np.expand_dims(threshold_mention_bounds, axis=0)
        threshold_scores = np.expand_dims(threshold_scores, axis=0)
        threshold_mention_scores = np.expand_dims(threshold_mention_scores, axis=0)
    # mention_scores = mention_scores[mention_scores > -float("inf")]
    threshold_entities_translate = {}
    pred_triples = []
    for m in range(len(threshold_entities)):
        mb = threshold_mention_bounds[m].tolist()
        mention_text = tokenizer.decode(example['tokens'][mb[0]-1:mb[1]])
        threshold_entities_translate[mention_text] = {
            "mention_idx": m, "candidate_entities": [],
            "scores": threshold_scores[m].tolist(),
            "mention_score": float(threshold_mention_scores[m])
        }
        pred_triples.append([str(threshold_entities[m][0]), mb[0]-1, mb[1]])
        for id in threshold_entities[m]:
            threshold_entities_translate[mention_text]["candidate_entities"].append(id2title[str(id)])
    new_ex = {
        "id": example["id"],
        "text": example["text"],
        "tokens": example["tokens"],
    }
    if "gold_triples" in example:
        all_pred_entities_pruned = []
        mention_masked_utterance = np.zeros(len(example['tokens']))
        # ensure well-formed-ness, prune overlaps
        # greedily pick highest scoring, then prune all overlapping
        for idx, mb in enumerate(pred_triples):
            if mb[1] >= len(mention_masked_utterance) or mb[2] >= len(mention_masked_utterance):
                continue
            # check if in existing mentions
            try:
                if sum(mention_masked_utterance[mb[1]:mb[2]]) > 0:
                    continue
            except:
                import pdb
                pdb.set_trace()
            all_pred_entities_pruned.append(mb)
            mention_masked_utterance[mb[1]:mb[2]] = 1
    else:
        all_pred_entities_pruned = pred_triples
        # all_pred_entities_pruned = []
        # mention_masked_utterance = np.zeros(len(example['tokens']))
        # ensure well-formed-ness, prune overlaps
        # greedily pick highest scoring, then prune all overlapping
        # for idx, mb in enumerate(pred_triples):
        #     if mb[1] >= len(mention_masked_utterance) or mb[2] > len(mention_masked_utterance):
        #         continue
        #     # check if in existing mentions
        #     try:
        #         if sum(mention_masked_utterance[mb[1]:mb[2]]) > 0:
        #             continue
        #     except:
        #         import pdb
        #         pdb.set_trace()
        #     all_pred_entities_pruned.append(mb)
        #     mention_masked_utterance[mb[1]:mb[2]] = 1
    new_ex['pred_mentions'] = threshold_entities_translate
    new_ex['pred_triples'] = all_pred_entities_pruned
    new_ex['pred_triples_string'] = [
        [id2title[triple[0]], tokenizer.decode(example['tokens'][triple[1]:triple[2]])]
        for triple in all_pred_entities_pruned
    ]
    # get scores
    if "gold_triples" in example:
        gold_triples = example["gold_triples"]
        new_ex["gold_triples"] = gold_triples
        num_overlap = entity_linking_tp_with_overlap(gold_triples, all_pred_entities_pruned)
        num_correct += num_overlap
        num_predicted += len(all_pred_entities_pruned)
        num_gold += len(gold_triples)
    new_examples.append(new_ex)

# compute metrics
p = num_correct / num_predicted
r = num_correct / num_gold
f1 = 2*p*r / (p+r)
print(f1)

# save
with open(os.path.join("/checkpoint/belindali/entity_link/data/{}/saved_preds".format(data), "{}_wiki_-5_no_overlaps.jsonl".format(split)), 'w') as wf:
    for new_ex in new_examples:
        b=wf.write(json.dumps(new_ex) + "\n")

