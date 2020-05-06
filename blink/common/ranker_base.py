# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch
from blink.common.span_extractors import batched_span_select, batched_index_select


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None,
        mention_aggregation_type=None, # options are None, "first", "avg"
    ):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        self.mention_aggregation_type = mention_aggregation_type
        self.tokens_to_aggregate = None
        self.aggregate_method = None
        if self.mention_aggregation_type is not None:
            self.mention_aggregation_type = self.mention_aggregation_type.split('_')
            self.tokens_to_aggregate = self.mention_aggregation_type[0]
            self.aggregate_method = self.mention_aggregation_type[1]
        self.dropout = nn.Dropout(0.1)
        # TODO!!! SOMETHING FOR FULL SPAN
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
        else:
            self.additional_linear = None

        if self.aggregate_method == 'linear':
            self.mention_agg_linear = nn.Linear(bert_output_dim, output_dim)
        elif self.aggregate_method == 'mlp':
            self.mention_agg_mlp = nn.Sequential(
                nn.Linear(bert_output_dim, bert_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1), 
                nn.Linear(bert_output_dim, output_dim),
            )
        else:
            self.mention_agg_mlp = None

    def forward(self, token_ids, segment_ids, attention_mask, mention_idxs=None):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        if self.mention_aggregation_type is not None:
            assert mention_idxs is not None
        # "'all_avg' to average across tokens in mention, 'fl_avg' to average across first/last tokens in mention, "
        # "'{all/fl}_linear' for linear layer over mention, '{all/fl}_mlp' to MLP over mention)",
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_pooler
        elif self.mention_aggregation_type is None:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_bert[:, 0, :]
        else:
            # try batched_span_select?
            if self.tokens_to_aggregate == 'all':
                (
                    embeddings,  # (batch_size, num_spans=1, max_batch_span_width, embedding_size)
                    mask,  # (batch_size, num_spans=1, max_batch_span_width)
                ) = batched_span_select(
                    output_bert,  # (batch_size, sequence_length, embedding_size)
                    mention_idxs.unsqueeze(1),  # (batch_size, num_spans=1, 2)
                )
                embeddings[~mask] = 0  # 0 out masked elements
                embeddings = embeddings.squeeze(1)
                mask = mask.squeeze(1)
                # embeddings = (batch_size, max_batch_span_width, embedding_size)
                if self.aggregate_method == 'avg':
                    embeddings = embeddings.sum(1) / mask.sum(1).float().unsqueeze(-1)
                    # embeddings = (batch_size, embedding_size)
            elif self.tokens_to_aggregate == 'fl':
                start_embeddings = batched_index_select(output_bert, mention_idxs[:,0])
                end_embeddings = batched_index_select(output_bert, mention_idxs[:,1])
                embeddings = torch.cat([start_embeddings.unsqueeze(1), end_embeddings.unsqueeze(1)], dim=1)
                # embeddings = (batch_size, 2, embedding_size)
                if self.aggregate_method == 'avg':
                    embeddings = embeddings.mean(1)
                    # embeddings = (batch_size, embedding_size)
                # TODO LINEAR/MLP
        
            # TODO AGGREGATE ACROSS DIMENSION 1!!!!
            if self.aggregate_method == 'linear':  # linear OR mlp
                # squeeze last 2 dimensions
                # TODO CHECK!!!
                # embeddings = embeddings.view(embeddings.size(0), -1)
                # embeddings = (batch_size, {seq_len/2} * embedding_size)
                embeddings = self.mention_agg_linear(self.dropout(embeddings))
            elif self.aggregate_method == 'mlp':
                # embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = self.mention_agg_mlp(self.dropout(embeddings))

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result


