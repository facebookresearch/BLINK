#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


if [ $# -le 0 ]
  then
    echo "Example usage:"
    echo "bash init_collection.sh --collection_name wikipedia_plus"
    echo "bash init_collection.sh --collection_name wikipedia_plus --add_sentence_data"
    exit 1
fi

NONRECOVERED=()

collection_name=""
add_sentence_data=""

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --collection_name)
    collection_name="$2"
    shift # past argument
    shift # past value
    ;;
    --add_sentence_data)
    add_sentence_data="--add_sentence_data"
    shift # past argument
    ;;
    *)    # unknown option
    NONRECOVERED+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

if [ -z "$collection_name" ]
then
    echo "Example usage:"
    echo "bash init_collection.sh --collection_name wikipedia_plus"
    echo "bash init_collection.sh --collection_name wikipedia_plus --add_sentence_data"
    exit 1
fi

echo "Non recovered parameters: $NONRECOVERED"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"num_incoming_links",
     "type":"plongs",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"num_tokens",
     "type":"plongs",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"title",
     "type":"text_general",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"aliases",
     "type":"text_general",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"desc",
     "type":"text_general",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"wikidata_desc",
     "type":"text_general",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field":{
     "name":"wikidata_id",
     "type":"text_general",
     "multiValued":false,
     "stored":true}
}' "http://localhost:8983/solr/$collection_name/schema"

if [ ! -z "$add_sentence_data" ]
then
    echo "Adding sentence data"
    for i in {1..10}
    do 
      key="sent_desc_$i"
      curl -X POST -H 'Content-type:application/json' --data-binary '{
        "add-field":{
          "name":'"$key"',
          "type":"text_general",
          "multiValued":false,
          "stored":true}
      }' "http://localhost:8983/solr/$collection_name/schema"
    done
fi





