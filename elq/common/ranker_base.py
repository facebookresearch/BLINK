# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None,
    ):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask, DEBUG=False):
        if DEBUG:
            import pdb
            pdb.set_trace()
        try:
            output_bert, output_pooler, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )
        except RuntimeError as e:
            print(token_ids.size())
            print(segment_ids.size())
            print(attention_mask.size())
            print(e)
            import pdb
            pdb.set_trace()
            output_bert, output_pooler, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )

        if self.additional_linear is not None:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_pooler
        else:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result
