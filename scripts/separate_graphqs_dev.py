import json
import os

all_docs = {}  # docid -> doc count

with open(
    "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{0}.entities.all_pos.filtered_on_all.no_partials.json".format(
        "train"
    )
) as f:
    f_parsed = json.load(f)
for q in f_parsed:
    doc_id = str(q['question_id'])[:3]
    if doc_id not in all_docs:
        all_docs[doc_id] = []
    all_docs[doc_id].append(q)


dev_docs = []
train_docs = []
num_dev_exs = 0

for doc in all_docs:
    if num_dev_exs >= 700:
        train_docs += all_docs[doc]
    else:
        num_dev_exs += len(all_docs[doc])
        dev_docs += all_docs[doc]


with open(
    "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{0}.entities.all_pos.filtered_on_all.split.json".format(
        "dev"
    ), "w"
) as wf:
    json.dump(dev_docs, wf)

with open(
    "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{0}.entities.all_pos.filtered_on_all.split.json".format(
        "train"
    ), "w"
) as wf:
    json.dump(train_docs, wf)

