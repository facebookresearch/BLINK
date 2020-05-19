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
from blink.common.span_extractors import batched_span_select, batched_index_select


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
    def __init__(self, bert_output_dim):
        super(MentionScoresHead, self).__init__()
        self.bound_classifier = nn.Linear(bert_output_dim, 2)

    def forward(self, bert_output, mask_ctxt):
        logits = self.bound_classifier(bert_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # impossible to choose masked tokens as starts/ends of spans
        start_logits[~mask_ctxt] = -float("Inf")
        end_logits[~mask_ctxt] = -float("Inf")
        return start_logits, end_logits


class GetContextEmbedsHead(nn.Module):
    def __init__(self, mention_aggregation_type, dropout=0.1):
        super(GetContextEmbedsHead, self).__init__()
        # for aggregating mention outputs of context encoder
        self.mention_aggregation_type = mention_aggregation_type.split('_')
        self.tokens_to_aggregate = self.mention_aggregation_type[0]
        self.aggregate_method = self.mention_aggregation_type[1]
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, bert_output, mention_idxs):
        # "'all_avg' to average across tokens in mention, 'fl_avg' to average across first/last tokens in mention, "
        # "'{all/fl}_linear' for linear layer over mention, '{all/fl}_mlp' to MLP over mention)",
        # get embedding of [CLS] token
        # try batched_span_select?
        if self.tokens_to_aggregate == 'all':
            try:
                (
                    embedding_ctxt,  # (batch_size, num_spans=1, max_batch_span_width, embedding_size)
                    mask,  # (batch_size, num_spans=1, max_batch_span_width)
                ) = batched_span_select(
                    bert_output,  # (batch_size, sequence_length, embedding_size)
                    mention_idxs.unsqueeze(1),  # (batch_size, num_spans=1, 2)
                )
            except:
                print(bert_output)
                print(mention_idxs)
                import pdb
                pdb.set_trace()
            embedding_ctxt[~mask] = 0  # 0 out masked elements
            embedding_ctxt = embedding_ctxt.squeeze(1)
            mask = mask.squeeze(1)
            # embedding_ctxt = (batch_size, max_batch_span_width, embedding_size)
            if self.aggregate_method == 'avg':
                embedding_ctxt = embedding_ctxt.sum(1) / mask.sum(1).float().unsqueeze(-1)
                # embedding_ctxt = (batch_size, embedding_size)
        elif self.tokens_to_aggregate == 'fl':
            start_embeddings = batched_index_select(bert_output, mention_idxs[:,0])
            end_embeddings = batched_index_select(bert_output, mention_idxs[:,1])
            embedding_ctxt = torch.cat([start_embeddings.unsqueeze(1), end_embeddings.unsqueeze(1)], dim=1)
            # embeddings = (batch_size, 2, embedding_size)
            if self.aggregate_method == 'avg':
                embedding_ctxt = embedding_ctxt.mean(1)
                # embeddings = (batch_size, embedding_size)
            # TODO LINEAR/MLP
    
        # TODO AGGREGATE ACROSS DIMENSION 1!!!!
        if self.aggregate_method == 'linear':  # linear OR mlp
            # squeeze last 2 dimensions
            # TODO CHECK!!!
            # embeddings = embeddings.view(embeddings.size(0), -1)
            # embeddings = (batch_size, {seq_len/2} * embedding_size)
            embedding_ctxt = self.mention_agg_linear(self.dropout(embedding_ctxt))
        elif self.aggregate_method == 'mlp':
            # embeddings = embeddings.view(embeddings.size(0), -1)
            embedding_ctxt = self.mention_agg_mlp(self.dropout(embedding_ctxt))
        return embedding_ctxt


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)
        cand_bert = BertModel.from_pretrained(params['bert_model'], output_hidden_states=True)
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

        bert_output_dim = ctxt_bert.embeddings.word_embeddings.weight.size(1)

        self.do_mention_detection = params.get('do_mention_detection', False)
        self.mention_aggregation_type = params.get('mention_aggregation_type', None)
        classification_heads_dict = {}
        if self.mention_aggregation_type is not None:
            classification_heads_dict['get_context_embeds'] = GetContextEmbedsHead(self.mention_aggregation_type)
            if self.do_mention_detection:
                classification_heads_dict['mention_scores'] = MentionScoresHead(bert_output_dim)
        self.classification_heads = nn.ModuleDict(classification_heads_dict)

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
        gold_mention_idxs=None,
    ):
        embedding_ctxt = None
        start_logits = None
        end_logits = None
        if token_idx_ctxt is not None:
            # get context encoding
            if self.mention_aggregation_type is None:
                # OLD system: don't do mention aggregation (use tokens around mention)
                embedding_ctxt = self.context_encoder(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt,# DEBUG=True,
                )
            else:
                # NEW system: aggregate mention tokens
                # retrieve mention tokens
                bert_output, _, _ = self.context_encoder.bert_model(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt,  # what is segment IDs?
                )

                # get start/end logits for scoring/backprop purposes
                if self.do_mention_detection:
                    # jointly model mention detection--get start/end logits (for scoring purposes)
                    start_logits, end_logits = self.classification_heads['mention_scores'](bert_output, mask_ctxt)
                    if gold_mention_idxs is None:
                        # use logit values to detect mention
                        # (to consider > 1 candidate, take N largest in 1st dimension)
                        start_pos = start_logits.argmax(1)
                        end_pos = end_logits.argmax(1)
                        # may be non-well-formed... (consider as incorrect)
                        non_well_formed_mask = end_pos < start_pos
                        start_pos[non_well_formed_mask] = -1
                        end_pos[non_well_formed_mask] = -1

                        mention_idxs = torch.stack([start_pos, end_pos]).t()
                    else:
                        # use gold mention
                        mention_idxs = gold_mention_idxs
                else:
                    # DON"T jointly model mention detection, mention bounds *must* be given
                    assert gold_mention_idxs is not None
                    mention_idxs = gold_mention_idxs

                try:
                    embedding_ctxt = self.classification_heads['get_context_embeds'](bert_output, mention_idxs)
                except:
                    print(token_idx_ctxt)
                    print(gold_mention_idxs)
                    import pdb
                    pdb.set_trace()

        embedding_cands = None
        if token_idx_cands is not None:
            # get candidate encoding
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands, start_logits, end_logits

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
            # if head_name == 'mention_scores':
            #     num_classes = state_dict[prefix + 'classification_heads.' + head_name + 'bound_classifier.weight'].size(0)
            #     inner_dim = state_dict[prefix + 'classification_heads.' + head_name + 'bound_classifier.bias'].size(0)

            # if getattr(self.args, 'load_checkpoint_heads', False):
            #     if head_name not in current_head_names:
            #         self.register_classification_head(head_name, num_classes, inner_dim)
            # else:
            if head_name not in current_head_names:
                print(
                    'WARNING: deleting classification head ({}) from checkpoint '
                    'not present in current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
            # elif (
            #     num_classes != self.classification_heads[head_name].bound_classifier.out_features
            #     or inner_dim != self.classification_heads[head_name].bound_classifier.out_features
            # ):
            #     print(
            #         'WARNING: deleting classification head ({}) from checkpoint '
            #         'with different dimensions than current model: {}'.format(head_name, k)
            #     )
            #     keys_to_delete.append(k)
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
                cand_enc_only=params.get("load_cand_enc_only", False)
            )
        self.loss_type = "mml"  # TODO make this a parameter
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False, cand_enc_only=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
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
 
    def encode_context(self, cands, gold_mention_idxs=None):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        try:
            embedding_context, _, start_logits, end_logits = self.model(
                token_idx_ctxt=token_idx_cands,
                segment_idx_ctxt=segment_idx_cands, mask_ctxt=mask_cands,
                token_idx_cands=None, segment_idx_cands=None, mask_cands=None,
                gold_mention_idxs=gold_mention_idxs,
            )
        except:
            print(token_idx_cands)
            print(segment_idx_cands)
            print(mask_cands)
            print(gold_mention_idxs)
            embedding_context, _, start_logits, end_logits = self.model(
                token_idx_ctxt=token_idx_cands,
                segment_idx_ctxt=segment_idx_cands, mask_ctxt=mask_cands,
                token_idx_cands=None, segment_idx_cands=None, mask_cands=None,
                gold_mention_idxs=gold_mention_idxs,
            )
            import pdb
            pdb.set_trace()
        return embedding_context, start_logits, end_logits

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands, _, _ = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        text_encs=None,  # pre-computed mention encoding.
        start_logits=None,  # pre-computed mention start logits
        end_logits=None,  # pre-computed mention end logits
        cand_encs=None,  # pre-computed candidate encoding.
        gold_mention_idxs=None,
    ):
        if text_encs is None or (
            (start_logits is None or end_logits is None) and ((
                hasattr(self.model, 'do_mention_detection') and self.model.do_mention_detection
            ) or (
                hasattr(self.model, 'module') and self.model.module.do_mention_detection
            ))
        ):
            # Encode contexts first
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
                text_vecs, self.NULL_IDX
            )
            embedding_ctxt, _, start_logits, end_logits = self.model(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
                None, None, None,
                gold_mention_idxs=gold_mention_idxs,
            )
        else:
            # Context encoding is given, do not need to re-compute
            embedding_ctxt = text_encs

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            if random_negs:
                # matmul across all cand_encs
                return embedding_ctxt.mm(cand_encs.t()), start_logits, end_logits
            else:
                embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
                cand_encs = cand_encs.unsqueeze(2)  # batchsize x embed_size x 1
                scores = torch.bmm(embedding_ctxt, cand_encs)  # batchsize x 1 x 1
                scores = torch.squeeze(scores)
                return scores, start_logits, end_logits

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands, _, _ = self.model(
            None, None, None,
            token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives (returns batchsize x batchsize of scores)
            return embedding_ctxt.mm(embedding_cands.t()), start_logits, end_logits
        else:
            # train on hard negatives (returns batchsize x 1 of scores)
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores, start_logits, end_logits

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(
        self, context_input, cand_input,
        cand_encs=None, 
        text_encs=None,  # pre-computed mention encoding.
        start_logits=None,  # pre-computed mention start logits
        end_logits=None,  # pre-computed mention end logits
        label_input=None,
        gold_mention_idxs=None, return_loss=True,
    ):
        # TODO MULTIPLE GOLD MENTIONS / SAMPLE
        # ONLY CALLED IF WE HAVE the lABELS
        flag = label_input is None
        scores, start_logits, end_logits = self.score_candidate(
            context_input, cand_input, random_negs=flag,
            cand_encs=cand_encs,# if flag else None,
            text_encs=text_encs,
            start_logits=start_logits, end_logits=end_logits,
            gold_mention_idxs=gold_mention_idxs,
        )

        if not return_loss:
            return scores, start_logits, end_logits

        span_loss = 0
        if start_logits is not None:
            N = context_input.size(0)  # batch size
            M = 1  # num_mentions per instance (just 1, so far) TODO remove hardcode
            span_loss = self.get_span_loss(
                gold_mention_idxs[:,0].unsqueeze(-1), gold_mention_idxs[:,1].unsqueeze(-1),
                start_logits, end_logits, N, M)

        bs = scores.size(0)
        # TODO modify target to *correct* label (pass in a label_input???)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            # log P(entity|mention) + log P(mention) = log [P(entity|mention)P(mention)]
            loss = F.cross_entropy(scores, target, reduction="mean") + span_loss
        else:
            # TODO: seems only training with 1 closest label?
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input.float()) + span_loss
        return loss, scores

    def get_span_loss(
        self, gold_start_positions, gold_end_positions,
        start_logits, end_logits, N, M,  # global_step,
    ):
        # clamp to range 0 <= position <= max positions (ignored_index)
        ignored_index = start_logits.size(1)
        gold_start_positions.clamp_(0, ignored_index)
        gold_end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignored_index)

        # compute mention start/end loss (unbind # of mentions dimension)
        start_losses = [loss_fct(start_logits, _start_positions) \
                        for _start_positions in torch.unbind(gold_start_positions, dim=1)]
        end_losses = [loss_fct(end_logits, _end_positions) \
                      for _end_positions in torch.unbind(gold_end_positions, dim=1)]
        # BS x num_mentions
        loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
            torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
        loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
        if self.loss_type == "mml":
            span_loss = self._take_mml(loss_tensor)
            # try:
            #     assert span_loss == loss_tensor.sum()
            # except AssertionError:
            #     import pdb
            #     pdb.set_trace()
        # elif self.loss_type == "hard-em":
        #     if numpy.random.random()<min(global_step/self.tau, 0.8):
        #         span_loss = self._take_min(loss_tensor)
        #     else:
        #         span_loss = self._take_mml(loss_tensor)
        else:
            raise NotImplementedError()
        return span_loss

    def _take_min(self, loss_tensor):
        return torch.sum(torch.min(
            loss_tensor + 2*torch.max(loss_tensor)*(loss_tensor==0).type(torch.FloatTensor).cuda(), 1)[0])

    def _take_mml(self, loss_tensor):
        # just sum if only 1 instance
        marginal_likelihood = torch.sum(torch.exp(
                - loss_tensor - 1e10 * (loss_tensor==0).float()), 1)
        return -torch.sum(torch.log(marginal_likelihood + \
                                    torch.ones(loss_tensor.size(0)).cuda()*(marginal_likelihood==0).float()))


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask