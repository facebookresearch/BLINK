# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import nltk.data
import argparse
import sys

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output", type=str, help="The full path to the data folder", required=True
)

args = parser.parse_args()

data_folder = args.output

output_file_name = "title2enriched_parsed_obj_plus.p"
output_file_path = os.path.join(data_folder, output_file_name)

if os.path.isfile(output_file_path):
    print("Output file `{}` already exists!".format(output_file_path))
    sys.exit()

print("Reading title2parsed_obj data")
title2enriched_parsed_obj_file_name = "title2enriched_parsed_obj.p"
title2enriched_parsed_obj_path = os.path.join(
    data_folder, title2enriched_parsed_obj_file_name
)
title2parsed_obj = pickle.load(open(title2enriched_parsed_obj_path, "rb"))

print("Reading title2parsed_obj_full_data")
title2parsed_obj_full_data_file_name = "title2parsed_obj_full_data.p"
title2parsed_obj_full_data_full_path = os.path.join(
    data_folder, title2parsed_obj_full_data_file_name
)
title2parsed_obj_full = pickle.load(open(title2parsed_obj_full_data_full_path, "rb"))

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

for title in tqdm(title2parsed_obj_full.keys()):
    lines = title2parsed_obj_full[title]["lines_full_text"][1:]  # remove title
    lines = [
        line for line in lines if not line.startswith("Section::")
    ]  # remove section titles
    lines = [
        line.strip() for line in lines if line != ""
    ]  # remove blank lines and trailing spaces
    text = " ".join(lines)

    sentences = sent_detector.tokenize(text)
    sentences = [sent.strip() for sent in sentences]

    for k in range(0, min(10, len(sentences))):
        key = "sent_desc_{}".format(k + 1)
        value = sentences[k]
        title2parsed_obj[title][key] = value

print("Dumping", output_file_path)
pickle.dump(title2parsed_obj, open(output_file_path, "wb"), protocol=4)

