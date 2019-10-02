# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import bz2
import sys
import pickle
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    help="The full path to the wikidata dump for processing",
    required=True,
)
parser.add_argument(
    "--output", type=str, help="The full path to the output folder", required=True
)

args = parser.parse_args()

input_file_path = args.input
output_file_path = args.output

if not os.path.isfile(input_file_path):
    print("Input file `{}` doesn't exist!".format(output_file_path))
    sys.exit()

if os.path.isfile(output_file_path):
    print("Output file `{}` already exists!".format(output_file_path))
    sys.exit()

id_title2parsed_obj = {}

num_lines = 0
with bz2.open(input_file_path, "rt") as f:
    for line in f:
        num_lines += 1

c = 0

with bz2.open(input_file_path, "rt") as f:
    for line in f:
        c += 1

        if c % 1000000 == 0:
            print("Processed: {:.2f}%".format(c * 100 / num_lines))

        try:
            json_obj = json.loads(line.strip().strip(","))

            if ("sitelinks" not in json_obj) or ("enwiki" not in json_obj["sitelinks"]):
                continue

            id_, title = json_obj["id"], json_obj["sitelinks"]["enwiki"]["title"]
            key = id_, title

            parsed_obj = {}

            if "en" in json_obj["aliases"]:
                parsed_obj["aliases"] = [
                    alias["value"] for alias in json_obj["aliases"]["en"]
                ]
            else:
                parsed_obj["aliases"] = None

            if "en" in json_obj["labels"]:
                parsed_obj["wikidata_label"] = json_obj["labels"]["en"]["value"]
            else:
                parsed_obj["wikidata_label"] = None

            if "en" in json_obj["descriptions"]:
                parsed_obj["description"] = json_obj["descriptions"]["en"]["value"]
            else:
                parsed_obj["description"] = None

            if "enwikiquote" in json_obj["sitelinks"]:
                parsed_obj["enwikiquote_title"] = json_obj["sitelinks"]["enwikiquote"][
                    "title"
                ]

            id_title2parsed_obj[key] = parsed_obj

        except Exception as e:
            line = line.strip().strip(",")

            if line == "[" or line == "]":
                continue

            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Exception:", exc_type, "- line", exc_tb.tb_lineno)
            if len(line) < 30:
                print("Failed line:", line)

print("Processed: {:.2f}%".format(c * 100 / num_lines))
print("Dumping", output_file_path)
pickle.dump(id_title2parsed_obj, open(output_file_path, "wb"), protocol=4)

