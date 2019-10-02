#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#



if [ $# -le 1 ]
  then
    echo "Usage: ./process_wikipedia_dump_links.sh wikipedia_xml_dump_path output_folder_path"
    exit 1
fi

xml_file_path=$1
parent_output_folder=$2

LIBS_PATH=libs

output_folder_extractor=${parent_output_folder}/en_extractor_links
mkdir -p $output_folder_extractor

full_wikipedia_data_file_path="${parent_output_folder}/en-wikilinks"

if [[ ! -f $full_wikipedia_data_file_path ]]; then
  # Process the raw wikipedia dump to remove the markup

  echo "Working on $(basename ${xml_file_path})"

  if [ ! -d "$LIBS_PATH/wikiextractor" ]
  then
      echo "Cloning wikiextractor..."
      mkdir -p $LIBS_PATH
      git clone https://github.com/attardi/wikiextractor.git $LIBS_PATH/wikiextractor
  fi

  $LIBS_PATH/wikiextractor/WikiExtractor.py --processes 40 -o ${output_folder_extractor} -q -s -l $xml_file_path


  # Merge the output of the processing (wikiextractor outputs many files) into one file

  c=0
  for small_file in $(find $output_folder_extractor -type f); 
  do	
    cat ${small_file} >> $full_wikipedia_data_file_path
    c=$((c + 1))
    echo "Processed $c files"
  done
  echo "Mark-up is removed"
else
  echo "$full_wikipedia_data_file_path already created"
fi

filtered_wikipedia_data_file_path=${parent_output_folder}/en-wikilinks-processed
python blink/candidate_retrieval/process_wiki_extractor_output_links.py --input $full_wikipedia_data_file_path --output $filtered_wikipedia_data_file_path
