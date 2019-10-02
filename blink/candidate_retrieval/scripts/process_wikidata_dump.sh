# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash


if [ $# -le 1 ]
  then
    echo "Usage: ./process_wikidata_dump.sh wikidata_json_dump_path data_folder_path"
    exit 1
fi

json_file_path=$1
data_folder_path=$2
output_file_path="$data_folder_path/wikidataid_title2parsed_obj.p"

# Extract information from wikidata dump 
python blink/candidate_retrieval/process_wikidata.py --input $json_file_path --output $output_file_path