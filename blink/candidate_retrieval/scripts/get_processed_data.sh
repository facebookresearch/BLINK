#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

if [ $# -le 0 ]
  then
    echo "Usage: ./get_processed_data.sh data_folder_path"
    exit 1
fi

data_folder_path=$1/KB_data
mkdir -p $data_folder_path

wikipedia_xml_dump="$data_folder_path/enwiki-pages-articles.xml.bz2"
wikidata_json_dump="$data_folder_path/wikidata-all.json.bz2"
en_index_precomputed="$data_folder_path/index_enwiki.db"

if [[ ! -f $wikipedia_xml_dump ]]; then
  echo "downloading $wikipedia_xml_dump"
  wget http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2 -O $wikipedia_xml_dump
fi

if [[ ! -f $wikidata_json_dump ]]; then
  echo "downloading $wikidata_json_dump"
  wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 -O $wikidata_json_dump
fi

if [[ ! -f $en_index_precomputed ]]; then
  echo "downloading $en_index_precomputed"
  wget https://public.ukp.informatik.tu-darmstadt.de/wikimapper/index_enwiki-20190420.db -O $en_index_precomputed
fi

echo "Processing wikipedia dump for text"
bash blink/candidate_retrieval/scripts/process_wikipedia_dump.sh $wikipedia_xml_dump $data_folder_path

echo "Processing wikipedia dump for links"
bash blink/candidate_retrieval/scripts/process_wikipedia_dump_links.sh $wikipedia_xml_dump $data_folder_path

echo "Processing wikidata dump"
bash blink/candidate_retrieval/scripts/process_wikidata_dump.sh $wikidata_json_dump $data_folder_path

echo "Linking wikipedia with wikidata"
bash blink/candidate_retrieval/scripts/link_wikipedia_and_wikidata.sh $en_index_precomputed $data_folder_path

echo "Enrich linked data with the number of tokens and number of incoming links"
python blink/candidate_retrieval/enrich_data.py --output $data_folder_path

echo "Create a dump that also contains the first 10 sentences numbered and given as separate fields"
python blink/candidate_retrieval/process_intro_sents.py --output $data_folder_path

echo "The data has been processed and dumped at the folder: ${data_folder_path}"


