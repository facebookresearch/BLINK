# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.biencoder.allennlp_span_utils import batched_span_select, batched_index_select
from blink.biencoder.utils import batch_reshape_mask_left


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


def get_submodel_from_state_dict(state_dict, prefix):
    # get only submodel specified with prefix 'prefix' from the state_dict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix)+1:]  # +1 for '.'  
            new_state_dict[key] = value
    return new_state_dict


class MentionScoresHead(nn.Module):
    def __init__(
        self, bert_output_dim, scoring_method="qa_linear", max_mention_length=10, 
    ):
        super(MentionScoresHead, self).__init__()
        self.scoring_method = scoring_method
        self.max_mention_length = max_mention_length
        if self.scoring_method == "qa_linear":
            self.bound_classifier = nn.Linear(bert_output_dim, 3)
        elif self.scoring_method == "qa_mlp" or self.scoring_method == "qa":  # for back-compatibility
            self.bound_classifier = nn.Sequential(
                nn.Linear(bert_output_dim, bert_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bert_output_dim, 3),
            )
        else:
            raise NotImplementedError()

    def forward(self, bert_output, mask_ctxt):
        '''
        Retuns scores for *inclusive* mention boundaries
        '''
        # (bs, seqlen, 3)
        logits = self.bound_classifier(bert_output)
        if self.scoring_method[:2] == "qa":
            # (bs, seqlen, 1); (bs, seqlen, 1); (bs, seqlen, 1)
            start_logprobs, end_logprobs, mention_logprobs = logits.split(1, dim=-1)
            # (bs, seqlen)
            start_logprobs = start_logprobs.squeeze(-1)
            end_logprobs = end_logprobs.squeeze(-1)
            mention_logprobs = mention_logprobs.squeeze(-1)
            # impossible to choose masked tokens as starts/ends of spans
            start_logprobs[~mask_ctxt] = -float("Inf")
            end_logprobs[~mask_ctxt] = -float("Inf")
            mention_logprobs[~mask_ctxt] = -float("Inf")

            # take sum of log softmaxes:
            # log p(mention) = log p(start_pos && end_pos) = log p(start_pos) + log p(end_pos)
            # DIM: (bs, starts, ends)
            mention_scores = start_logprobs.unsqueeze(2) + end_logprobs.unsqueeze(1)
            # do this in naive way, because I can't figure out a better way...
            # (bs, starts, ends)
            mention_cum_scores = torch.zeros(mention_scores.size(), dtype=mention_scores.dtype).to(mention_scores.device)
            # add ends
            mention_logprobs_end_cumsum = torch.zeros(mask_ctxt.size(0), dtype=mention_scores.dtype).to(mention_scores.device)
            for i in range(mask_ctxt.size(1)):
                mention_logprobs_end_cumsum += mention_logprobs[:,i]
                mention_cum_scores[:,:,i] += mention_logprobs_end_cumsum.unsqueeze(-1)
            # subtract starts
            mention_logprobs_start_cumsum = torch.zeros(mask_ctxt.size(0), dtype=mention_scores.dtype).to(mention_scores.device)
            for i in range(mask_ctxt.size(1)-1):
                mention_logprobs_start_cumsum += mention_logprobs[:,i]
                mention_cum_scores[:,(i+1),:] -= mention_logprobs_start_cumsum.unsqueeze(-1)

            # DIM: (bs, starts, ends)
            mention_scores += mention_cum_scores

            # DIM: (starts, ends, 2) -- tuples of [start_idx, end_idx]
            mention_bounds = torch.stack([
                # torch.arange(mention_scores.size(0)).unsqueeze(-1).unsqueeze(-1).expand_as(mention_scores),  # index in batch
                torch.arange(mention_scores.size(1)).unsqueeze(-1).expand(mention_scores.size(1), mention_scores.size(2)),  # start idxs
                torch.arange(mention_scores.size(1)).unsqueeze(0).expand(mention_scores.size(1), mention_scores.size(2)),  # end idxs
            ], dim=-1).to(mask_ctxt.device)
            # DIM: (starts, ends)
            mention_sizes = mention_bounds[:,:,1] - mention_bounds[:,:,0] + 1  # (+1 as ends are inclusive)

            # Remove invalids (startpos > endpos, endpos > seqlen) and renormalize
            # DIM: (bs, starts, ends)
            valid_mask = (mention_sizes.unsqueeze(0) > 0) & mask_ctxt.unsqueeze(1)
            # DIM: (bs, starts, ends)
            mention_scores[~valid_mask] = -float("inf")  # invalids have logprob=-inf (p=0)
            # DIM: (bs, starts * ends)
            mention_scores = mention_scores.view(mention_scores.size(0), -1)
            # DIM: (bs, starts * ends, 2)
            mention_bounds = mention_bounds.view(-1, 2)
            mention_bounds = mention_bounds.unsqueeze(0).expand(mention_scores.size(0), mention_scores.size(1), 2)
        
        if self.max_mention_length is not None:
            mention_scores, mention_bounds = self.filter_by_mention_size(
                mention_scores, mention_bounds, self.max_mention_length,
            )

        return mention_scores, mention_bounds
    
    def filter_by_mention_size(self, mention_scores, mention_bounds, max_mention_length):
        '''
        Filter all mentions > maximum mention length
        mention_scores: torch.FloatTensor (bsz, num_mentions)
        mention_bounds: torch.LongTensor (bsz, num_mentions, 2)
        '''
        # (bsz, num_mentions)
        mention_bounds_mask = (mention_bounds[:,:,1] - mention_bounds[:,:,0] <= max_mention_length)
        # (bsz, num_filtered_mentions)
        mention_scores = mention_scores[mention_bounds_mask]
        mention_scores = mention_scores.view(mention_bounds_mask.size(0),-1)
        # (bsz, num_filtered_mentions, 2)
        mention_bounds = mention_bounds[mention_bounds_mask]
        mention_bounds = mention_bounds.view(mention_bounds_mask.size(0),-1,2)
        return mention_scores, mention_bounds


class GetContextEmbedsHead(nn.Module):
    def __init__(self, mention_aggregation_type, ctxt_output_dim, cand_output_dim, dropout=0.1):
        """
        mention_aggregation_type
            `all_avg`: average across tokens in mention
            `fl_avg`: to average across first/last tokens in mention
            `{all/fl}_linear`: for linear layer over mention
            `{all/fl}_mlp` to MLP over mention
        """
        super(GetContextEmbedsHead, self).__init__()
        # for aggregating mention outputs of context encoder
        self.mention_aggregation_type = mention_aggregation_type.split('_')
        self.tokens_to_aggregate = self.mention_aggregation_type[0]
        self.aggregate_method = "_".join(self.mention_aggregation_type[1:])
        self.dropout = nn.Dropout(dropout)
        if self.mention_aggregation_type == 'all_avg' or self.mention_aggregation_type == 'none':
            assert ctxt_output_dim == cand_output_dim

        if self.aggregate_method == 'linear':
            self.mention_agg_linear = nn.Linear(ctxt_output_dim * 2, cand_output_dim)
        elif self.aggregate_method == 'avg_linear':
            self.mention_agg_linear = nn.Linear(ctxt_output_dim, cand_output_dim)
        elif self.aggregate_method == 'mlp':
            self.mention_agg_mlp = nn.Sequential(
                nn.Linear(bert_output_dim, bert_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1), 
                nn.Linear(bert_output_dim, output_dim),
            )
        else:
            self.mention_agg_mlp = None

    def forward(self, bert_output, mention_bounds):
        '''
        bert_output
            (bs, seqlen, embed_dim)
        mention_bounds: both bounds are inclusive [start, end]
            (bs, num_spans, 2)
        '''
        # get embedding of [CLS] token
        if mention_bounds.size(0) == 0:
            return mention_bounds
        if self.tokens_to_aggregate == 'all':
            (
                embedding_ctxt,  # (batch_size, num_spans, max_batch_span_width, embedding_size)
                mask,  # (batch_size, num_spans, max_batch_span_width)
            ) = batched_span_select(
                bert_output,  # (batch_size, sequence_length, embedding_size)
                mention_bounds,  # (batch_size, num_spans, 2)
            )
            embedding_ctxt[~mask] = 0  # 0 out masked elements
            # embedding_ctxt = (batch_size, num_spans, max_batch_span_width, embedding_size)
            if self.aggregate_method.startswith('avg'):
                embedding_ctxt = embedding_ctxt.sum(2) / mask.sum(2).float().unsqueeze(-1)
                # embedding_ctxt = (batch_size, num_spans, embedding_size)
            if self.aggregate_method == 'avg_linear':
                embedding_ctxt = self.mention_agg_linear(embedding_ctxt)
                # embedding_ctxt = (batch_size, num_spans, output_dim)
        elif self.tokens_to_aggregate == 'fl':
            # assert 
            start_embeddings = batched_index_select(bert_output, mention_bounds[:,:,0])
            end_embeddings = batched_index_select(bert_output, mention_bounds[:,:,1])
            embedding_ctxt = torch.cat([start_embeddings.unsqueeze(2), end_embeddings.unsqueeze(2)], dim=2)
            # embedding_ctxt = (batch_size, num_spans, 2, embedding_size)
            if self.aggregate_method == 'avg':
                embedding_ctxt = embedding_ctxt.mean(2)
                # embedding_ctxt = (batch_size, num_spans, embedding_size)
            elif self.aggregate_method == 'linear':
                embedding_ctxt = embedding_ctxt.view(embedding_ctxt.size(0), embedding_ctxt.size(1), -1)
                # embedding_ctxt = (batch_size, num_spans, 2 * embedding_size)
                embedding_ctxt = self.mention_agg_linear(embedding_ctxt)
                # embedding_ctxt = (batch_size, num_spans, output_dim)
        else:
            raise NotImplementedError()
    
        return embedding_ctxt


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)
        if params["load_cand_enc_only"]:
            bert_model = "bert-large-uncased"
        else:
            bert_model = params['bert_model']
        cand_bert = BertModel.from_pretrained(
            bert_model,
            output_hidden_states=True,
        )
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        if params.get("freeze_cand_enc", False):
            for param in self.cand_encoder.parameters():
                param.requires_grad = False

        self.config = ctxt_bert.config

        ctxt_bert_output_dim = ctxt_bert.embeddings.word_embeddings.weight.size(1)

        self.do_mention_detection = params.get('do_mention_detection', False)
        self.mention_aggregation_type = params.get('mention_aggregation_type', None)
        self.classification_heads = nn.ModuleDict({})
        self.linear_compression = None
        if self.mention_aggregation_type is not None:
            classification_heads_dict = {'get_context_embeds': GetContextEmbedsHead(
                self.mention_aggregation_type,
                ctxt_bert_output_dim,
                cand_bert.embeddings.word_embeddings.weight.size(1),
            )}
            if self.do_mention_detection:
                classification_heads_dict['mention_scores'] = MentionScoresHead(
                    ctxt_bert_output_dim,
                    params["mention_scoring_method"],
                    params.get("max_mention_length", 10),
                )
            self.classification_heads = nn.ModuleDict(classification_heads_dict)
        elif ctxt_bert_output_dim != cand_bert.embeddings.word_embeddings.weight.size(1):
            # mapping to make the output dimensions match for dot-product similarity
            self.linear_compression = nn.Linear(ctxt_bert_output_dim, cand_bert.embeddings.word_embeddings.weight.size(1))

    def get_raw_ctxt_encoding(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
    ):
        """
            Gets raw, shared context embeddings from BERT,
            to be used by both mention detector and entity linker

        Returns:
            torch.FloatTensor (bsz, seqlen, embed_dim)
        """
        assert self.mention_aggregation_type is not None
        raw_ctxt_encoding, _, _ = self.context_encoder.bert_model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
        )
        return raw_ctxt_encoding

    def get_ctxt_mention_scores(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        raw_ctxt_encoding = None,
    ):
        """
            Gets mention scores using raw context encodings

        Inputs:
            raw_ctxt_encoding: torch.FloatTensor (bsz, seqlen, embed_dim)
        Returns:
            torch.FloatTensor (bsz, num_total_mentions): mention scores/logits
            torch.IntTensor (bsz, num_total_mentions): mention boundaries
        """
        # (bsz, seqlen, embed_dim)
        if raw_ctxt_encoding is not None:
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

        # (num_total_mentions,); (num_total_mentions,)
        return self.classification_heads['mention_scores'](
            raw_ctxt_encoding, mask_ctxt,
        )

    def prune_ctxt_mentions(
        self,
        mention_logits,
        mention_bounds,
        num_cand_mentions,
        threshold,
    ):
        '''
            Prunes mentions based on mention scores/logits (by either
            `threshold` or `num_cand_mentions`, whichever yields less candidates)

        Inputs:
            mention_logits: torch.FloatTensor (bsz, num_total_mentions)
            mention_bounds: torch.IntTensor (bsz, num_total_mentions)
            num_cand_mentions: int
            threshold: float
        Returns:
            torch.FloatTensor(bsz, max_num_pred_mentions): top mention scores/logits
            torch.IntTensor(bsz, max_num_pred_mentions, 2): top mention boundaries
            torch.BoolTensor(bsz, max_num_pred_mentions): mask on top mentions
            torch.BoolTensor(bsz, total_possible_mentions): mask for reshaping from total possible mentions -> max # pred mentions
        '''
        # (bsz, num_cand_mentions); (bsz, num_cand_mentions)
        top_mention_logits, mention_pos = mention_logits.topk(num_cand_mentions, sorted=True)
        # (bsz, num_cand_mentions, 2)
        #   [:,:,0]: index of batch
        #   [:,:,1]: index into top mention in mention_bounds
        mention_pos = torch.stack([torch.arange(mention_pos.size(0)).to(mention_pos.device).unsqueeze(-1).expand_as(mention_pos), mention_pos], dim=-1)
        # (bsz, num_cand_mentions)
        top_mention_pos_mask = torch.sigmoid(top_mention_logits).log() > threshold
        # (total_possible_mentions, 2)
        #   tuples of [index of batch, index into mention_bounds] of what mentions to include
        mention_pos = mention_pos[top_mention_pos_mask | (
            # 2nd part of OR: if nothing is > threshold, use topK that are > -inf
            ((top_mention_pos_mask.sum(1) == 0).unsqueeze(-1)) & (top_mention_logits > -float("inf"))
        )]
        mention_pos = mention_pos.view(-1, 2)
        # (bsz, total_possible_mentions)
        #   mask of possible logits
        mention_pos_mask = torch.zeros(mention_logits.size(), dtype=torch.bool).to(mention_pos.device)
        mention_pos_mask[mention_pos[:,0], mention_pos[:,1]] = 1
	# (bsz, max_num_pred_mentions, 2)
        chosen_mention_bounds, chosen_mention_mask = batch_reshape_mask_left(mention_bounds, mention_pos_mask, pad_idx=0)
        # (bsz, max_num_pred_mentions)
        chosen_mention_logits, _ = batch_reshape_mask_left(mention_logits, mention_pos_mask, pad_idx=-float("inf"), left_align_mask=chosen_mention_mask)
        return chosen_mention_logits, chosen_mention_bounds, chosen_mention_mask, mention_pos_mask

    def get_ctxt_embeds(
        self,
        raw_ctxt_encoding,
        mention_bounds,
    ):
        """
            Get candidate scores + embeddings associated with passed-in mention_bounds

        Input
            raw_ctxt_encoding: torch.FloatTensor (bsz, seqlen, embed_dim)
                shared embeddings straight from BERT
            mention_bounds: torch.IntTensor (bsz, max_num_pred_mentions, 2)
                top mention boundaries

        Returns
            torch.FloatTensor (bsz, max_num_pred_mentions, embed_dim)
        """
        import pdb
        pdb.set_trace()
        # (bs, max_num_pred_mentions, embed_dim)
        embedding_ctxt = self.classification_heads['get_context_embeds'](raw_ctxt_encoding, mention_bounds)
        if self.linear_compression is not None:
            embedding_ctxt = self.linear_compression(embedding_ctxt)
        return embedding_ctxt

    def forward_ctxt(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        gold_mention_bounds=None,
        gold_mention_bound_mask=None,
        num_cand_mentions=50,
        topK_threshold=-4.5,
    ):
        """
        If gold_mention_bounds is set, returns mention embeddings of passed-in mention bounds
        Otherwise, uses top-scoring mentions
        """
        if self.mention_aggregation_type is None:
            '''
            OLD system: don't do mention aggregation (use tokens around mention)
            '''
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )
            # linear mapping to correct context length
            if self.linear_compression is not None:
                embedding_ctxt = self.linear_compression(embedding_ctxt)
            return embedding_ctxt, None, None, None

        else:
            '''
            NEW system: aggregate mention tokens
            '''
            # (bs, seqlen, embed_size)
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

            top_mention_bounds = None
            top_mention_logits = None
            if self.do_mention_detection:
                mention_logits, mention_bounds = self.get_ctxt_mention_scores(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt, raw_ctxt_encoding,
                )
                if gold_mention_bounds is None:
                    (
                        top_mention_logits, top_mention_bounds, top_mention_mask, all_mention_mask,
                    ) = self.prune_ctxt_mentions(
                        mention_logits, mention_bounds, num_cand_mentions, topK_threshold,
                    )

            if top_mention_bounds is None:
                # use gold mention
                top_mention_bounds = gold_mention_bounds
                top_mention_mask = gold_mention_bound_mask

            assert top_mention_bounds is not None

            # (bs, num_pred_mentions OR num_gold_mentions, embed_size)
            embedding_ctxt = self.get_ctxt_embeds(
                raw_ctxt_encoding, top_mention_bounds,
            )
            # for merging dataparallel, only 1st dimension can differ...
            return (
                embedding_ctxt.view(-1, embedding_ctxt.size(-1)),
                all_mention_mask,
                mention_logits, mention_bounds,
                torch.tensor([embedding_ctxt.size(1)]).to(embedding_ctxt.device),
            )

    def forward_candidate(
        self,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        return self.cand_encoder(
            token_idx_cands, segment_idx_cands, mask_cands
        )

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        num_cand_mentions=50,
        topK_threshold=-4.5,
    ):
        """
        If gold_mention_bounds is set, returns mention embeddings of passed-in mention bounds
        Otherwise, uses top-scoring mentions
        """
        embedding_ctxt = None
        all_mention_logits = None
        all_mention_bounds = None
        mention_mask = None
        embedding_cands = None
        if token_idx_ctxt is not None:
            (
                embedding_ctxt, top_mention_mask,
                all_mention_logits, all_mention_bounds,
                max_num_pred_mentions,
            ) = self.forward_ctxt(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
                gold_mention_bounds, gold_mention_bounds_mask, num_cand_mentions, topK_threshold,
            )
        if token_idx_cands is not None:
            embedding_cands = self.forward_candidate(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return (
            embedding_ctxt, embedding_cands, top_mention_mask,
            all_mention_logits, all_mention_bounds,
            max_num_pred_mentions,
        )

    def upgrade_state_dict_named(self, state_dict):
        prefix = ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            if head_name not in current_head_names:
                print(
                    'WARNING: deleting classification head ({}) from checkpoint '
                    'not present in current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(
                model_path,
                cand_enc_only=params.get("load_cand_enc_only", False),
            )
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False, cand_enc_only=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        if cand_enc_only:
            cand_state_dict = get_submodel_from_state_dict(state_dict, 'cand_encoder')
            self.model.cand_encoder.load_state_dict(cand_state_dict)
        else:
            self.model.upgrade_state_dict_named(state_dict)
            self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(
        self, cands, gold_mention_bounds=None, gold_mention_bounds_mask=None,
        num_cand_mentions=50, topK_threshold=-4.5,
    ):
        """
        if gold_mention_bounds specified, selects according to gold_mention_bounds,
        otherwise selects according to top-scoring mentions
        """
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        (
            embedding_context, _,
            mention_logits, mention_bounds,
            mention_pos_mask,
        ) = self.model(
            token_idx_cands, segment_idx_cands, mask_cands,
            None, None, None,
            gold_mention_bounds=gold_mention_bounds,
            num_cand_mentions=num_cand_mentions,
            topK_threshold=topK_threshold,
        )
        # reshape
        import pdb
        pdb.set_trace()
        # (bsz, max_num_pred_mentions, 2)
        chosen_mention_bounds, chosen_mention_mask = batch_reshape_mask_left(mention_bounds, mention_pos_mask, pad_idx=0)
        # (bsz, max_num_pred_mentions)
        chosen_mention_logits, _ = batch_reshape_mask_left(mention_logits, mention_pos_mask, pad_idx=-float("inf"), left_align_mask=chosen_mention_mask)

        #top_mention_mask = top_mention_mask.view(, , -1)
        # concatenated across 0th dimension
        return (
            embedding_context, chosen_mention_mask,
            chosen_mention_logits, chosen_mention_bounds,
            mention_logits, mention_bounds,
        )

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands, _, _, _ = self.model(
            None, None, None,
            token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_vecs is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        text_encs=None,  # pre-computed mention encoding.
        mention_logits=None,  # pre-computed mention logits
        mention_bounds=None,
        cand_encs=None,  # pre-computed candidate encoding.
        gold_mention_bounds=None,
        gold_mention_bound_mask=None,
        all_inputs_mask=None,
        topK_threshold=-4.5,
    ):
        if not random_negs:
            assert all_inputs_mask is not None
        if text_encs is None or (
            (mention_logits is None and mention_bounds is not None) and ((
                hasattr(self.model, 'do_mention_detection') and self.model.do_mention_detection
            ) or (
                hasattr(self.model, 'module') and self.model.module.do_mention_detection
            ))
        ):
            # embedding_ctxt: (bs, num_gold_mentions/num_pred_mentions, embed_size)
            (
                embedding_context, top_mention_mask,
                top_mention_logits, top_mention_bounds,
                all_mention_logits, all_mention_bounds,
            ) = self.encode_context(
                text_vecs, gold_mention_bounds=gold_mention_bounds,
            )
            if gold_mention_bounds is not None:
                # (bs * num_mentions, embed_size)
                embedding_ctxt = embedding_ctxt.view(-1, embedding_ctxt.size(-1))
        else:
            # Context encoding is given, do not need to re-compute
            embedding_ctxt = text_encs

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            if gold_mention_bounds is not None:
                if embedding_ctxt.size(1) != gold_mention_bounds.size(1):
                    # (bs, #_gold_mentions, embed_dim)
                    try:
                        embedding_ctxt = self.model.module.classification_heads['get_context_embeds'](embedding_ctxt, gold_mention_bounds)
                    except:
                        embedding_ctxt = self.model.classification_heads['get_context_embeds'](embedding_ctxt, gold_mention_bounds)
                if all_inputs_mask is None:
                    all_inputs_mask = gold_mention_bound_mask
                # (bs * num_mentions, embed_size)
                embedding_ctxt = embedding_ctxt.view(-1, embedding_ctxt.size(-1))[all_inputs_mask.flatten()]
                # candidates and embedding contexts correspond
            else:
                # TODO
                import pdb
                pdb.set_trace()
            if random_negs:
                # matmul across all cand_encs
                return embedding_ctxt.mm(cand_encs.t()), mention_logits, mention_bounds
            else:
                # (bs * num_mentions, embed_size)
                cand_encs = cand_encs.view(-1, cand_encs.size(-1))[all_inputs_mask.flatten()]
                embedding_ctxt = embedding_ctxt.unsqueeze(1)  # (batchsize * num_mentions) x 1 x embed_size
                cand_encs = cand_encs.unsqueeze(2)  # (batchsize * num_mentions) x embed_size x 1
                scores = torch.bmm(embedding_ctxt, cand_encs)  # (batchsize * num_mentions) x 1 x 1
                scores = torch.squeeze(scores)
                return scores, mention_logits, mention_bounds

        # cand_vecs: (bs, num_gold_mentions, 1, cand_width) -> (bs * num_gold_mentions, cand_width)
        cand_vecs = cand_vecs.view(-1, cand_vecs.size(-2), cand_vecs.size(-1)).squeeze(1)
        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        # embedding_cands: (bs * num_gold_mentions, embed_dim)
        _, embedding_cands, _, _, _ = self.model(
            None, None, None,
            token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # if gold_mention_bounds is not None:
            if all_inputs_mask is None:
                all_inputs_mask = gold_mention_bound_mask
            # (bs x num_tot_mentions, embed_size)
            embedding_ctxt = embedding_ctxt.view(-1, embedding_ctxt.size(-1))
            # train on random negatives (returns bs*num_mentions x bs*num_mentions of scores)
            all_scores = embedding_ctxt[all_inputs_mask.flatten()].mm(embedding_cands[all_inputs_mask.flatten()].t())
            # scores_mask = all_inputs_mask.flatten() & all_inputs_mask.flatten().unsqueeze(-1)
            # all_scores[~scores_mask] = -float("inf")
            return all_scores, mention_logits, mention_bounds
        else:
            # train on hard negatives (returns batchsize x 1 of scores)
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores, mention_logits, mention_bounds

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(
        self, context_input, cand_input,
        cand_encs=None, 
        text_encs=None,  # pre-computed mention encoding.
        mention_logits=None,  # pre-computed mention logits
        mention_bounds=None,
        label_input=None,
        gold_mention_bounds=None,
        gold_mention_bound_mask=None,
        all_inputs_mask=None,  # should be non-none if we are using negs
        return_loss=True,
    ):
        flag = label_input is None

        scores, mention_logits, mention_bounds = self.score_candidate(
            context_input, cand_input, random_negs=flag,
            cand_encs=cand_encs,
            text_encs=text_encs,
            mention_logits=mention_logits,
            mention_bounds=mention_bounds,
            gold_mention_bounds=gold_mention_bounds,
            gold_mention_bound_mask=gold_mention_bound_mask,
            all_inputs_mask=all_inputs_mask,
        )

        if not return_loss:
            return scores, mention_logits, mention_bounds

        '''
        COMPUTE MENTION LOSS
        '''
        span_loss = 0
        if mention_logits is not None and mention_bounds is not None:
            N = context_input.size(0)  # batch size
            M = gold_mention_bounds.size(1)  # num_mentions per instance (just 1, so far)
            # 1 value
            span_loss = self.get_span_loss(
                gold_mention_bounds=gold_mention_bounds, 
                gold_mention_bound_mask=gold_mention_bound_mask,
                mention_logits=mention_logits, mention_bounds=mention_bounds,
                N=N, M=M,
            )

        '''
        COMPUTE EL LOSS
        '''
        if label_input is None:
            '''
            Hard negatives (negatives passed in)
            '''
            # scores: (bs*num_mentions [filtered], bs*num_mentions [filtered])
            target = torch.LongTensor(torch.arange(scores.size(1)))
            target = target.to(self.device)
            # log P(entity|mention) + log P(mention) = log [P(entity|mention)P(mention)]
            loss = F.cross_entropy(scores, target, reduction="mean") + span_loss
        else:
            '''
            Random negatives (use in-batch negatives)
            '''
            # scores: (bs, num_spans, all_embeds)
            if flag:
                all_inputs_mask = gold_mention_bound_mask
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # scores: ([masked] bs * num_spans), label_input: ([masked] bs * num_spans)
            label_input = label_input.flatten()[all_inputs_mask.flatten()]
            loss = loss_fct(scores, label_input.float()) + span_loss

        return loss, scores

    def get_span_loss(
        self, gold_mention_bounds, gold_mention_bound_mask, mention_logits, mention_bounds, N, M,  # global_step,
    ):
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")

        gold_mention_bounds[~gold_mention_bound_mask] = -1  # ensure don't select masked to score
        # triples of [ex in batch, mention_idx in gold_mention_bounds, idx in mention_bounds]
        # use 1st, 2nd to index into gold_mention_bounds, 1st, 3rd to index into mention_bounds
        gold_mention_pos_idx = ((
            mention_bounds.unsqueeze(1) - gold_mention_bounds.unsqueeze(2)  # (bs, num_mentions, start_pos * end_pos, 2)
        ).abs().sum(-1) == 0).nonzero()
        # gold_mention_pos_idx should have 1 entry per masked element
        # (num_gold_mentions [~gold_mention_bound_mask])
        gold_mention_pos = gold_mention_pos_idx[:,2]

        # (bs, total_possible_spans)
        gold_mention_binary = torch.zeros(mention_logits.size(), dtype=mention_logits.dtype).to(gold_mention_bounds.device)
        gold_mention_binary[gold_mention_pos_idx[:,0], gold_mention_pos_idx[:,2]] = 1

        # prune masked spans
        mask = mention_logits != -float("inf")
        masked_mention_logits = mention_logits[mask]
        masked_gold_mention_binary = gold_mention_binary[mask]

        # (bs, total_possible_spans)
        span_loss = loss_fct(masked_mention_logits, masked_gold_mention_binary)

        return span_loss


def to_bert_input(token_idx, null_idx):
    """
    token_idx is a 2D tensor int.
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
