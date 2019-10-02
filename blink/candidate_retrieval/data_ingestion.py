# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import pysolr
import pickle
import emoji
import time
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--processed_data_file_path",
    type=str,
    help="The full path to the data file",
    required=True,
)

parser.add_argument(
    "--collection_name",
    type=str,
    help="The solr collection name, in which the ingestion should be performed",
    required=True,
)

parser.add_argument(
    "--add_sentence_data", dest="add_sentence_data", action="store_true"
)
parser.set_defaults(add_sentence_data=False)

parser.add_argument(
    "--remove_disambiguation_pages",
    dest="remove_disambiguation_pages",
    action="store_true",
)
parser.set_defaults(remove_disambiguation_pages=False)

parser.add_argument("--min_tokens", type=int, default=0)

args = parser.parse_args()
processed_data_path = args.processed_data_file_path
collection_name = args.collection_name

# processed_data_path = "/scratch/martinjosifoski/data/en-wiki-filtered-wikidata"


def remove_all_docs():
    solr.delete(q="*:*")


def load_data():
    return pickle.load(open(processed_data_path, "rb"))


def get_data_for_key(data, title):
    obj = {}

    obj["id"] = data[title]["wikipedia_id"]
    obj["title"] = title

    if ("wikidata_info" in data[title]) and (
        data[title]["wikidata_info"]["wikidata_id"] is not None
    ):
        obj["wikidata_id"] = data[title]["wikidata_info"]["wikidata_id"]
    else:
        obj["wikidata_id"] = data[title]["wikidata_id_from_index"]

    description = data[title]["intro_concatenated"]
    obj["desc"] = description

    if "wikidata_info" in data[title]:
        if "description" in data[title]["wikidata_info"]:
            wikidata_description = data[title]["wikidata_info"]["description"]
        else:
            wikidata_description = ""

        if ("aliases" in data[title]["wikidata_info"]) and (
            data[title]["wikidata_info"]["aliases"]
        ) is not None:
            aliases = " ".join(
                [
                    '"{}"'.format(alias)
                    for alias in data[title]["wikidata_info"]["aliases"]
                    if alias not in emoji.UNICODE_EMOJI
                ]
            )
        else:
            aliases = ""
    else:
        aliases = ""
        wikidata_description = ""

    obj["aliases"] = aliases
    obj["wikidata_desc"] = wikidata_description
    obj["num_tokens"] = data[title]["num_tokens"]
    obj["num_incoming_links"] = data[title].get("num_incoming_links", 0)

    if args.add_sentence_data:
        for k in range(0, 10):
            key = "sent_desc_{}".format(k + 1)
            obj[key] = data[title].get(key, "")

    return obj


print("Loading data")
title2data = load_data()

for key in title2data:
    title2data[key]["intro_concatenated"] = " ".join(
        [line for line in title2data[key]["intro_lines"] if line != ""]
    )

# Filter documents with less then `args.min_tokens` tokens
if args.min_tokens != 0:
    print("Removing documents with less then {} tokens".format(args.min_tokens))
    print("Number of docs BEFORE removal:", len(title2data))
    title2data = {
        key: value
        for key, value in title2data.items()
        if value["num_tokens"] >= args.min_tokens
    }
    print("Number of docs AFTER removal:", len(title2data))
    print("")

# Remove disambiguation pages
if args.remove_disambiguation_pages:
    print("Remove disambiguation pages")
    print("Number of docs BEFORE removal:", len(title2data))
    titles_to_delete = []

    for title in title2data:
        parsed_obj = title2data[title]
        if ("disambiguation" in title) or ("Disambiguation" in title):
            titles_to_delete.append(title)
        else:
            if (parsed_obj.get("wikidata_info", None) is not None) and (
                parsed_obj["wikidata_info"].get("description", None) is not None
            ):
                wikidata_info = parsed_obj["wikidata_info"]
                if ("disambiguation page" in wikidata_info["description"]) or (
                    "Disambiguation page" in wikidata_info["description"]
                ):
                    titles_to_delete.append(title)

    for title in titles_to_delete:
        del title2data[title]

    print("Number of docs AFTER removal:", len(title2data))
    print("Number of removed docs:", len(titles_to_delete))
    print("")

ingestion_data = [get_data_for_key(title2data, key) for key in title2data]

print("Starting ingestion")
wall_start = time.time()

l = 0
r = step = 10000
solr = pysolr.Solr(
    "http://localhost:8983/solr/{}".format(collection_name),
    always_commit=True,
    timeout=100,
)

c = 0

for r in range(r, len(ingestion_data), step):
    c += 1

    if (c % 10) == 0:
        print("Processed", c, "batches")

    temp_data = ingestion_data[l:r]
    solr.add(temp_data, commit=True)
    l = r

solr.add(ingestion_data[l : len(ingestion_data)], commit=True)
solr.commit()

print("The processing took:", (time.time() - wall_start) / 60, " minutes")

