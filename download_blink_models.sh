#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

if [[ ! -f biencoder_wiki_large.bin ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin
fi

if [[ ! -f biencoder_wiki_large.json ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json
fi

if [[ ! -f entity.jsonl ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/entity.jsonl
fi

if [[ ! -f all_entities_large.t7 ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7
fi

if [[ ! -f crossencoder_wiki_large.bin ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin
fi

if [[ ! -f crossencoder_wiki_large.json ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.json
fi

cd "$ROOD_DIR"
