#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


if [ $# -le 1 ]
  then
    echo "Example usage:"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus --min_tokens 20 --add_sentence_data --remove_disambiguation_pages"
    exit 1
fi

NONRECOVERED=()

processed_data_file_path=""
collection_name=""
collection_name_only=""
min_tokens=""
add_sentence_data=""
remove_disambiguation_pages=""

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --processed_data_file_path)
    processed_data_file_path="--processed_data_file_path $2"
    shift # past argument
    shift # past value
    ;;
    --collection_name)
    collection_name="--collection_name $2"
    collection_name_only="$2"
    shift # past argument
    shift # past value
    ;;
    --min_tokens)
    min_tokens="--min_tokens $2"
    shift # past argument
    shift # past value
    ;;
    --add_sentence_data)
    add_sentence_data="--add_sentence_data"
    shift # past argument
    ;;
    --remove_disambiguation_pages)
    remove_disambiguation_pages="--remove_disambiguation_pages"
    shift # past argument
    ;;
    *)    # unknown option
    NONRECOVERED+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

if [ -z "$collection_name_only" ]
then
    echo "Example usage:"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus --min_tokens 20 --add_sentence_data --remove_disambiguation_pages"
    exit 1
fi

if [ -z "$processed_data_file_path" ]
then
    echo "Example usage:"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus"
    echo "bash ingestion_data.sh --processed_data_file_path /scratch/martinjosifoski/data/title2enriched_parsed_obj_plus.p --collection_name wikipedia_plus --min_tokens 20 --add_sentence_data --remove_disambiguation_pages"
    exit 1
fi

echo "Non recovered parameters: $NONRECOVERED"


options_data_ingestion="$processed_data_file_path $collection_name $add_sentence_data $remove_disambiguation_pages $min_tokens"
options_init_collection="$collection_name $add_sentence_data"

# echo "=====Creating collection with name '$collection_name_only'====="
# sudo su - solr -c "/opt/solr/bin/solr create -c $collection_name_only -n data_driven_schema_configs"

echo "=====Initializing collection====="
bash blink/candidate_retrieval/scripts/init_collection.sh $options_init_collection

echo "=====Populating collection====="
python blink/candidate_retrieval/data_ingestion.py $options_data_ingestion