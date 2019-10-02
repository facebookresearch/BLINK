# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import os
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_folder",
    type=str,
    help="The full path to the output folder",
    required=True,
)

args = parser.parse_args()

output_folder = args.output_folder
output_file_path = os.path.join(output_folder, "en-wiki-filtered-wikidata")
if os.path.isfile(output_file_path):
    print("Output file `{}` already exists!".format(output_file_path))
    sys.exit()

# Add wikidata_id from the download index to wikipedia articles whenever we have it
wikipediaid2wikidataid_file_path = os.path.join(
    output_folder, "wikipediaid2wikidataid.p"
)
wikipedia_data_filtered_file_path = os.path.join(output_folder, "en-wiki-filtered")

wikipediaid2wikidataid = pickle.load(open(wikipediaid2wikidataid_file_path, "rb"))
wikipedia_data_filtered = pickle.load(open(wikipedia_data_filtered_file_path, "rb"))

for key in wikipedia_data_filtered.keys():
    wikipedia_id, wikipedia_title = key
    wikipedia_id = int(wikipedia_id)
    if wikipedia_id in wikipediaid2wikidataid:
        wikidata_id = wikipediaid2wikidataid[wikipedia_id]
        wikipedia_data_filtered[key]["wikidata_id_from_index"] = wikidata_id
    else:
        wikipedia_data_filtered[key]["wikidata_id_from_index"] = None


# Read the processed wikidata object and generate amenable mappings
wikidataid_title2parsed_obj_file_path = os.path.join(
    output_folder, "wikidataid_title2parsed_obj.p"
)
wikidataid_title2parsed_obj = pickle.load(
    open(wikidataid_title2parsed_obj_file_path, "rb")
)

title2parsed_obj = {}
wikidataid2parsed_obj = {}

for key in wikidataid_title2parsed_obj.keys():
    wikidata_id, wikipedia_title = key

    wikidataid_title2parsed_obj[key]["wikidata_id"] = wikidata_id
    wikidataid_title2parsed_obj[key]["wikipedia_title"] = wikipedia_title

    title2parsed_obj[wikipedia_title] = wikidataid_title2parsed_obj[key]
    wikidataid2parsed_obj[wikidata_id] = wikidataid_title2parsed_obj[key]

    matched_by_title = 0
    not_matched_by_title_list = []

    matched_by_id = 0
    not_matched_by_anything = []


# link wikipedia with wikidata
for key in wikipedia_data_filtered.keys():
    wikipedia_id, wikipedia_title = key
    wikipedia_id = int(wikipedia_id)
    wikidata_id_from_index = wikipedia_data_filtered[key]["wikidata_id_from_index"]

    ## 1) TITLE 2) ID
    ## works better, linking is more accurate
    if wikipedia_title in title2parsed_obj:
        matched_by_title += 1
        wikipedia_data_filtered[key]["wikidata_info"] = title2parsed_obj[
            wikipedia_title
        ]
    else:
        not_matched_by_title_list.append(
            (wikipedia_id, wikipedia_title, wikidata_id_from_index)
        )
        if (wikidata_id_from_index is not None) and (
            wikidata_id_from_index in wikidataid2parsed_obj
        ):
            matched_by_id += 1
            wikipedia_data_filtered[key]["wikidata_info"] = wikidataid2parsed_obj[
                wikidata_id_from_index
            ]
        else:
            not_matched_by_anything.append(
                (wikipedia_id, wikipedia_title, wikidata_id_from_index)
            )

## 1) ID 2) TITLE
#     if (wikidata_id_from_index is not None) and (wikidata_id_from_index in wikidataid2parsed_obj):
#             matched_by_id += 1
#             wikipedia_data_filtered[key]['wikidata_info'] = wikidataid2parsed_obj[wikidata_id_from_index]
#     else:
#         not_matched_by_title_list.append((wikipedia_id, wikipedia_title, wikidata_id_from_index))
#         if wikipedia_title in title2parsed_obj:
#             matched_by_title += 1
#             wikipedia_data_filtered[key]['wikidata_info'] = title2parsed_obj[wikipedia_title]
#         else:
#             not_matched_by_anything.append((wikipedia_id, wikipedia_title, wikidata_id_from_index))


print("Matched by title:", matched_by_title)
print("Matched by id:", matched_by_id)
print("Not found:", len(not_matched_by_anything))

print("Dumping", output_file_path)
pickle.dump(wikipedia_data_filtered, open(output_file_path, "wb"), protocol=4)

