# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import pickle
import json
import emoji
import sys
import os
import io

import blink.candidate_retrieval.utils as utils

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--processed_mention_data_file_path",
    type=str,
    help="The full path to the mention data file",
    default="data/mention_dumps/train_and_eval_data",
)


parser.add_argument(
    "--dump_folder_path",
    type=str,
    help="The path to the dump folder",
    default="data/train_and_benchmark_processed_json",
)

# Keep pregenerated candidates
parser.add_argument(
    "--keep_pregenerated_candidates",
    action="store_true",
    help="Whether to keep the candidates given with the dataset.",
)


args = parser.parse_args()
print(args)

dump_folder = args.dump_folder_path
path_to_processed_mention_data = args.processed_mention_data_file_path
os.makedirs(dump_folder, exist_ok=True)
print("Reading data")
run_dump = pickle.load(open(path_to_processed_mention_data, "rb"))

mentions = run_dump["mentions"]

dataset2processed_mentions = {}

for m in tqdm(mentions):
    mention_obj = {}

    mention_obj["candidates"] = m["generated_candidates"]

    # Gold data
    mention_obj["gold_pos"] = m["gold_pos"]
    mention_obj["gold"] = m["gold"]

    # Mention data
    mention_obj["text"] = m["mention_orig"]
    # mention_obj['query_context_50'] = m['query_context_orig']
    # mention_obj['query_context_sent_prev_curr_next'] = utils.get_sent_context(m, 'prev_next', solr_escaped=False)

    # mention_obj['tagged_context_50'] = (m['left_context_orig'], m['right_context_orig'])
    prev_sent = m["sent_context_orig"][0] if m["sent_context_orig"][0] != None else ""
    next_sent = m["sent_context_orig"][2] if m["sent_context_orig"][2] != None else ""
    mention_obj["tagged_query_context_sent_prev_curr_next"] = (
        "{} {}".format(prev_sent, m["left_query_sent_context_orig"]).strip(),
        "{} {}".format(m["right_query_sent_context_orig"], next_sent).strip(),
    )
    mention_obj["tagged_query_context_sent_curr"] = (
        m["left_query_sent_context_orig"].strip(),
        m["right_query_sent_context_orig"].strip(),
    )

    # Keep the candidates given with the dataset (used for the purposes of comparison with baseline)
    if args.keep_pregenerated_candidates:
        mention_obj["pregenerated_candidates"] = m["candidates_data"]
        mention_obj["pregenerated_gold_pos"] = m["pre_gen_candidates_gold_pos"]

    # Add data to output dics
    dataset_name = m["dataset_name"]

    processed_mentions = dataset2processed_mentions.get(dataset_name, [])
    processed_mentions.append(mention_obj)

    dataset2processed_mentions[dataset_name] = processed_mentions

for dataset_name in dataset2processed_mentions:
    print("Dumping dataset:", dataset_name)
    processed_mentions = dataset2processed_mentions[dataset_name]

    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(dump_folder, file_name)

    # with open(txt_file_path, "w+") as file:
    with io.open(txt_file_path, mode="w", encoding="utf-8") as file:
        for idx, mention in enumerate(processed_mentions):
            json_string = json.dumps(mention)
            file.write(json_string)

            if idx != (len(processed_mentions) - 1):
                file.write("\n")

