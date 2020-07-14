import json
import pdb
import os

all_ids = {}
q_count = {}
shared = {}

name_list = ["belinda", "scott", "sewon", "srini"]

for doc in ["train_webqsp", "dev_webqsp", "test_webqsp", "train_graphqs", "test_graphqs"]:
    all_ids[doc] = {}
    for name in ["belinda", "scott", "sewon", "srini"]:
        all_ids[doc][name] = {}
        dataset = json.load(open("to_annotate_mention_bound/annotation_set_{0}/annotated/{1}.json".format(name, doc)))
        if name != "sewon":
            dataset = {d[0]: d[1] for d in dataset}
        else:
            dataset = {d["id"]: d for d in dataset}
        if os.path.exists("to_annotate_mention_bound/annotation_set_{0}/{1}.json".format(name, doc)):
            orig_dataset = json.load(open("to_annotate_mention_bound/annotation_set_{0}/{1}.json".format(name, doc)))
            orig_dataset = {d[0]: d[1] for d in orig_dataset}
        else:
            orig_dataset = dataset
        for q_id in orig_dataset:
            # for i, _q in enumerate(dataset):
            if q_id not in dataset:
                assert name == 'srini' or name == 'belinda'
                q_annot = orig_dataset[q_id]["question"]  # no mention bounds
            else:
                q_annot = dataset[q_id]
                if '[' not in q_annot["question"] and (name == "belinda" or name == "srini"):
                    q_annot["question"] = q_annot["question_with_hypothesis"]
                q_annot = q_annot["question"]
            all_ids[doc][name].update({q_id: q_annot})
            if doc + "_" + q_id not in q_count:
                q_count[doc + "_" + q_id] = []
            q_count[doc + "_" + q_id].append(name)


for _q in q_count:
    if len(q_count[_q]) > 1:
        _q_split = _q.split("_")
        doc = "_".join(_q_split[:-1])
        q_id = _q_split[-1]
        if len(q_count[_q]) == 2 and "srini" in q_count[_q] and "sewon" in q_count[_q]:
            shared[_q] = [all_ids[doc][name][q_id] for name in ["srini", "sewon"]]
        else:
            try:
                assert len(q_count[_q]) == 4
            except:
                pdb.set_trace()
                continue
        try:
            shared[_q] = [all_ids[doc][name][q_id] for name in q_count[_q]]
        except:
            pdb.set_traec()


shared_dataset_counts = {}
for _q in shared:
    if "_".join(_q.split("_")[:-1]) not in shared_dataset_counts:
        shared_dataset_counts["_".join(_q.split("_")[:-1])] = 0
    shared_dataset_counts["_".join(_q.split("_")[:-1])] += 1


num_shared_agree = {}
num_all_agree = 0
for _q in shared:
    num_shared_agree[_q] = 0
    if len(shared[_q]) < 4:
        for i in range(len(shared[_q])):
            if '[' not in shared[_q][i]:
                num_shared_agree[_q] += 1
            else:
                pdb.set_trace()
        if num_shared_agree[_q] == len(shared[_q]):
            num_all_agree += 1
            continue
    _q_set = set(shared[_q])
    if len(_q_set) == 1:
        num_shared_agree[_q] += 4
        num_all_agree += 1
    else:
        pdb.set_trace()
        # {'who was michael jackson in the [wiz]?', 'who was michael jackson in [the wiz]?'} -- 2,2
    # elif len(_q_set) == 2:
    #     num_shared_agree[_q] += 3
    # elif len(_q_set) == 3:
    #     num_shared_agree[_q] += 1
    # elif len(_q_set) == 4:
    #     num_shared_agree[_q] += 0
