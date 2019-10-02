#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


if [ $# -le 0 ]
  then
    echo "Usage: ./ingestion_wrapper.sh processed_data_folder"
    exit 1
fi

data_folder=$1
processed_data_file_path="$data_folder/KB_data/title2enriched_parsed_obj_plus.p"

# Create the collecitons (requires sudo access!)
# sudo bash create_solr_collections.sh wikipedia_plus wikipedia_plus_no_dis wikipedia_plus_no_dis_min_20 wikipedia_plus_no_dis_min_40
sudo bash blink/candidate_retrieval/scripts/create_solr_collections.sh wikipedia

# Enriched + sentence data
bash blink/candidate_retrieval/scripts/ingest_data.sh --processed_data_file_path $processed_data_file_path --collection_name wikipedia --add_sentence_data

# # Prev + removal of disambiguation pages
# bash ingest_data.sh --processed_data_file_path $processed_data_file_path --collection_name wikipedia_plus_no_dis --add_sentence_data --remove_disambiguation_pages

# # Prev + removal of docs with less then 20 words in total
# bash ingest_data.sh --processed_data_file_path $processed_data_file_path --collection_name wikipedia_plus_no_dis_min_20 --add_sentence_data --remove_disambiguation_pages --min_tokens 20

# # Prev + removal of docs with less then 40 words in total
# bash ingest_data.sh --processed_data_file_path $processed_data_file_path --collection_name wikipedia_plus_no_dis_min_40 --add_sentence_data --remove_disambiguation_pages --min_tokens 40