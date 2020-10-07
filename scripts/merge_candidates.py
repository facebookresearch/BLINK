# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import os

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--path_to_saved_chunks', type=str, required=True, help='filepath to directory containing saved chunks')
parser.add_argument('--chunk_size', type=int, default=1000000, help='size of each chunk')
args = parser.parse_args()

CHUNK_SIZES = args.chunk_size

all_chunks = []

for fn in range(0, 5903526, CHUNK_SIZES):
    f_chunk = os.path.join(
        args.path_to_saved_chunks, '{}_{}.t7'.format(fn, fn+CHUNK_SIZES),
    )
    if not os.path.exists(f_chunk) or os.path.getsize(f_chunk) == 0:
        continue
    loaded_chunk = torch.load(f_chunk)
    all_chunks.append(loaded_chunk[:CHUNK_SIZES])

all_chunks = torch.cat(all_chunks, dim=0)
torch.save(all_chunks, os.path.join(
    args.path_to_saved_chunks, 'all.t7'.format(fn, fn+CHUNK_SIZES),
))
