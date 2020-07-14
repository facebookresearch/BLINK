# TODO!!!!!
import json

with open("/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.train.entities.all_pos.json") as f:
    graph_trains = json.load(f)

graph_trains_qids = []
for ex in graph_trains:
    graph_trains_qids.append(ex['question_id'])

with open("/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.train.entities.json") as f:
    graph_trains_full = json.load(f)

for ex in graph_trains_full:
    if ex["question_id"] not in graph_trains_qids:
        # assert len([ent for ent in ex['entities'] if ent is not None]) == 0
        print(ex)

