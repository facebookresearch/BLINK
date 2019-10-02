# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sqlite3
import pickle
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    type=str,
    help="The full path to the precomputed index",
    required=True,
)
parser.add_argument(
    "--output_folder",
    type=str,
    help="The full path to the output folder",
    required=True,
)

args = parser.parse_args()

precomp_index_path = args.input_file
output_folder_path = args.output_folder

output_file = os.path.join(output_folder_path, "linktitle2wikidataid.p")

if not os.path.isfile(output_file):
    conn = sqlite3.connect(precomp_index_path)
    cursorObj = conn.cursor()
    cursorObj.execute("SELECT wikipedia_title, wikidata_id FROM mapping")
    data = cursorObj.fetchall()

    linktitle2wikidataid = {item[0]: item[1] for item in data}

    pickle.dump(linktitle2wikidataid, open(output_file, "wb"))
else:
    print("Output file `{}` already exists!".format(output_file))

output_file = os.path.join(output_folder_path, "wikipediaid2wikidataid.p")

if not os.path.isfile(output_file):
    conn = sqlite3.connect(precomp_index_path)
    cursorObj = conn.cursor()
    cursorObj.execute("SELECT wikipedia_id, wikidata_id FROM mapping")
    data = cursorObj.fetchall()

    wikipediaid2wikidataid = {item[0]: item[1] for item in data}

    pickle.dump(wikipediaid2wikidataid, open(output_file, "wb"))
else:
    print("Output file `{}` already exists!".format(output_file))
