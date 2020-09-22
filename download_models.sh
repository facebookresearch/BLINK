#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


set -e
set -u

ROOT_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOT_DIR/models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

if [[ ! -f elq_wiki_large.bin ]]; then
    wget http://dl.fbaipublicfiles.com/elq/elq_wiki_large.bin
fi

if [[ ! -f elq_webqsp_large.bin ]]; then
    wget http://dl.fbaipublicfiles.com/elq/elq_webqsp_large.bin
fi

if [[ ! -f elq_large_params.txt ]]; then
    wget http://dl.fbaipublicfiles.com/elq/elq_large_params.txt
fi

if [[ ! -f entity.jsonl ]]; then
    wget http://dl.fbaipublicfiles.com/elq/entity.jsonl
fi

if [[ ! -f entity_token_ids_128.t7 ]]; then
    wget http://dl.fbaipublicfiles.com/elq/entity_token_ids_128.t7
fi

if [[ ! -f all_entities_large.t7 ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7
fi

if [[ ! -f faiss_hnsw_index.pkl ]]; then
    wget http://dl.fbaipublicfiles.com/elq/faiss_hnsw_index.pkl
fi

cd "$ROOT_DIR"
