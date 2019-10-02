# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sqlite3
import pickle
import os
import io
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output", type=str, help="The full path to the data folder", required=True
)

args = parser.parse_args()

data_folder = args.output

output_file_name = "title2enriched_parsed_obj.p"
output_file_path = os.path.join(data_folder, output_file_name)

if os.path.isfile(output_file_path):
    print("Output file `{}` already exists!".format(output_file_path))
    sys.exit()

linktitle2wikidataid_file_name = "linktitle2wikidataid.p"
linktitle2wikidataid_path = os.path.join(data_folder, linktitle2wikidataid_file_name)
linktitle2wikidataid = pickle.load(open(linktitle2wikidataid_path, "rb"))

# Read links data
links_file_name = "en-wikilinks-processed"
links_file_path = os.path.join(data_folder, links_file_name)
links_data = pickle.load(open(links_file_path, "rb"))

print("Links data is loaded")

# Read full text data
full_num_tokens_file_name = "en-wiki-full-text"
full_num_tokens_file_path = os.path.join(data_folder, full_num_tokens_file_name)
full_num_tokens_data = pickle.load(open(full_num_tokens_file_path, "rb"))

print("Full text and number of tokens data is loaded")

# Read linked (wikipedia with wikidata) data
filtered_and_wikidata_file_name = "en-wiki-filtered-wikidata"
filtered_and_wikidata_file_path = os.path.join(
    data_folder, filtered_and_wikidata_file_name
)
filtered_and_wikidata_data = pickle.load(open(filtered_and_wikidata_file_path, "rb"))

print("Introduction text, linked with wikidata data is loaded")

# Transform the linked data into a title2parsed_obj dictionary
# Add the number of tokens information
title2parsed_obj = {}

for key in filtered_and_wikidata_data.keys():
    wikipedia_id, wikipedia_title = key

    filtered_and_wikidata_data[key]["wikipedia_id"] = wikipedia_id
    filtered_and_wikidata_data[key]["wikipedia_title"] = wikipedia_title
    filtered_and_wikidata_data[key]["num_tokens"] = full_num_tokens_data[key][
        "num_tokens"
    ]

    title2parsed_obj[wikipedia_title] = filtered_and_wikidata_data[key]


total = {"xml": 0, "regex": 0}
found = {"xml": 0, "regex": 0}
not_found = {"xml": [], "regex": []}

# Counting using the title
for key in links_data.keys():
    wikipedia_id, wikipedia_title = key

    if links_data[key]["links_xml"] != None:
        links = links_data[key]["links_xml"]
        total["xml"] = total["xml"] + len(links)

        for link in links:
            title = link["href_unquoted"]

            if title in title2parsed_obj:
                title2parsed_obj[title]["num_incoming_links"] = (
                    title2parsed_obj[title].get("num_incoming_links", 0) + 1
                )
                found["xml"] = found["xml"] + 1
            else:
                not_found["xml"].append(link)
    else:
        links = links_data[key]["links_regex"]
        total["regex"] = total["regex"] + len(links)

        for link in links:
            title = link["href_unquoted"]

            if title in title2parsed_obj:
                title2parsed_obj[title]["num_incoming_links"] = (
                    title2parsed_obj[title].get("num_incoming_links", 0) + 1
                )
                found["regex"] = found["regex"] + 1
            else:
                not_found["regex"].append(link)

print(
    "Matched {:2f}% using only the title".format(
        (found["xml"] + found["regex"]) * 100 / (total["xml"] + total["regex"])
    )
)

# Counting using the index
wikidataid2count = {}

for link in not_found["xml"] + not_found["regex"]:
    title = link["href_unquoted"]
    title = title.replace(" ", "_")

    if title in linktitle2wikidataid:
        wikidata_id = linktitle2wikidataid[title]
        wikidataid2count[wikidata_id] = wikidataid2count.get(wikidata_id, 0) + 1

        found["xml"] = found["xml"] + 1

    elif title.capitalize() in linktitle2wikidataid:
        wikidata_id = linktitle2wikidataid[title.capitalize()]
        wikidataid2count[wikidata_id] = wikidataid2count.get(wikidata_id, 0) + 1

        found["xml"] = found["xml"] + 1

print(
    "Matched {:2f}% by additionally using the title to wikidataid index".format(
        (found["xml"] + found["regex"]) * 100 / (total["xml"] + total["regex"])
    )
)

# Adding the counts from the index to the original dictionary
updated = 0
wikdiata_info = 0
wikidata_id_from_index = 0

for key in title2parsed_obj:
    parsed_obj = title2parsed_obj[key]
    wikidata_id = None

    if parsed_obj.get("wikidata_info", None) is not None:
        wikdiata_info += 1
        if parsed_obj["wikidata_info"].get("wikidata_id", None) is not None:
            wikidata_id = parsed_obj["wikidata_info"]["wikidata_id"]
    else:
        if parsed_obj.get("wikidata_id_from_index", None) is not None:
            wikidata_id_from_index += 1
            wikidata_id = parsed_obj["wikidata_id_from_index"]

    if (wikidata_id is not None) and (wikidata_id in wikidataid2count):
        parsed_obj["num_incoming_links"] = (
            parsed_obj.get("num_incoming_links", 0) + wikidataid2count[wikidata_id]
        )
        updated += 1

print("Dumping", output_file_path)
pickle.dump(title2parsed_obj, open(output_file_path, "wb"), protocol=4)

# Include unprocessed data and dump it together with the processed data
# (convenient if we want to extent the data that we use)
for wikipedia_title in title2parsed_obj.keys():
    wikipedia_id = title2parsed_obj[wikipedia_title]["wikipedia_id"]
    key = wikipedia_id, wikipedia_title

    title2parsed_obj[wikipedia_title]["links_data"] = {
        "links_xml": links_data[key]["links_xml"],
        "links_regex": links_data[key]["links_regex"],
    }

    title2parsed_obj[wikipedia_title]["lines_full_text"] = full_num_tokens_data[key][
        "lines"
    ]

output_file_name = "title2parsed_obj_full_data.p"
output_file_path = os.path.join(data_folder, output_file_name)

print("Dumping", output_file_path)
pickle.dump(title2parsed_obj, open(output_file_path, "wb"), protocol=4)

