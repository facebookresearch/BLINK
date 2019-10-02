#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


data_folder_path='data/KB_data'
en_index_precomputed="$data_folder_path/index_enwiki.db"

mkdir -p $data_folder_path

if [[ ! -f $en_index_precomputed ]]; then
  echo "downloading $en_index_precomputed"
  wget https://public.ukp.informatik.tu-darmstadt.de/wikimapper/index_enwiki-20190420.db -O $en_index_precomputed
fi

# Generate mappings between wikipedia and wikidata ids, and titles and wikidata ids
python blink/candidate_retrieval/generate_wiki2wikidata_mappings.py --input_file $en_index_precomputed --output_folder $data_folder_path