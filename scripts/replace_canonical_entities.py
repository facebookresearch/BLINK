import json
import os
import numpy as np
import torch
from elq.vcg_utils.measures import entity_linking_tp_with_overlap
from tqdm import tqdm

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data = "nq"
split= "dev"

# save
with open(os.path.join("/checkpoint/belindali/entity_link/data/{}/saved_preds".format(data), "{}.jsonl".format(split))) as f:
    examples = f.readlines()
    examples = [json.loads(line) for line in examples]

with open(os.path.join("/private/home/belindali/bart-closed-book-qa/data/", "{}open-{}.json".format(data, split))) as f:
    qa_examples = json.load(f)

num_same = 0
for example in examples:
    mb = example["pred_triples"]
    tokens = example["tokens"]
    for triple in range(len(mb)):
        if example["pred_triples_string"][triple][0].lower() == example["pred_triples_string"][triple][1]:
            num_same += 1
    # TODO compute string versions
    # text = tokenizer.decode(example["tokens"][:mb[0]]) + 

# 63316 / 79168
# 6958 / 8757

with open(os.path.join("/private/home/belindali/bart-closed-book-qa/data/", "{}open-{}.json".format(data, split))) as f:
    qa_examples = json.dump(new_examples, f)
