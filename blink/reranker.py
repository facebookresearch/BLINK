# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from blink.candidate_ranking.bert_reranking import BertReranker


def get_model(params):
    return BertReranker(params)
