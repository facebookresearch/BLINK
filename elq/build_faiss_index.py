# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import logging
import numpy
import os
import time
import torch

from elq.index.faiss_indexer import DenseFlatIndexer, DenseIVFFlatIndexer, DenseHNSWFlatIndexer
import elq.candidate_ranking.utils as utils

logger = utils.get_logger()

def main(params): 
    output_path = params["output_path"]

    logger.info("Loading candidate encoding from path: %s" % params["candidate_encoding"])
    candidate_encoding = torch.load(params["candidate_encoding"])
    vector_size = candidate_encoding.size(1)
    index_buffer = params["index_buffer"]
    if params["faiss_index"] == "hnsw":
        logger.info("Using HNSW index in FAISS")
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    elif params["faiss_index"] == "ivfflat":
        logger.info("Using IVF Flat index in FAISS")
        index = DenseIVFFlatIndexer(vector_size, 75, 100)
    else:
        logger.info("Using Flat index in FAISS")
        index = DenseFlatIndexer(vector_size, index_buffer)

    logger.info("Building index.")
    index.index_data(candidate_encoding.numpy())
    logger.info("Done indexing data.")

    if params.get("save_index", None):
        index.serialize(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="output file path",
    )
    parser.add_argument(
        "--candidate_encoding",
        default="models/all_entities_large.t7",
        type=str,
        help="file path for candidte encoding.",
    )
    parser.add_argument(
        "--faiss_index", type=str, choices=["hnsw", "flat", "ivfflat"],
        help='Which faiss index to use',
    )
    parser.add_argument(
        "--save_index", action='store_true', 
        help='If enabled, save index',
    )
    parser.add_argument(
        '--index_buffer', type=int, default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )

    params = parser.parse_args()
    params = params.__dict__

    main(params)
