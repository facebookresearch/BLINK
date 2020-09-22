# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
FAISS-based index components. Original from 
https://github.com/facebookresearch/DPR/blob/master/dpr/indexer/faiss_indexers.py
"""

import os
import logging
import pickle

import faiss
import numpy as np

logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        logger.info("Serializing index to %s", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )


# DenseFlatIndexer does exact search
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int = 1, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: np.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d", n)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes


# DenseIVFFlatIndexer does bucketed exact search
class DenseIVFFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int = 1, nprobe: int = 10, nlist: int = 100):
        super(DenseIVFFlatIndexer, self).__init__()
        self.nprobe = nprobe
        self.nlist = nlist
        quantizer = faiss.IndexFlatL2(vector_sz)  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, vector_sz, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe

    def index_data(self, data: np.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        self.index.train(data)
        self.index.add(data)
        logger.info("Total data indexed %d", n)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        index = faiss.IndexHNSWFlat(vector_sz, store_n, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index

    def index_data(self, data: np.array):
        n = len(data)

        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        self.index.add(data)
        logger.info("Total data indexed %d" % n)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1