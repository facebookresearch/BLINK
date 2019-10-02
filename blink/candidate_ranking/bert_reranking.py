# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import numpy as np

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class BertForReranking(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``
    
            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForReranking, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        entity_mask=None,
    ):
        num_choices = input_ids.shape[1]

        # from batch_size x cands x tokens -> (batch_size x cands) x tokens
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        flat_attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )

        flat_position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )

        outputs = self.bert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        entity_mask = (1.0 - entity_mask) * -1000.0
        reshaped_logits = reshaped_logits + entity_mask

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs


class BertReranker:
    def __init__(self, parameters):
        if "path_to_model" not in parameters:
            parameters["path_to_model"] = parameters["bert_model"]

        self.parameters = parameters

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not parameters["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # Load the fine-tuned model and the tokenizer used by it
        self.model = BertReranker.get_model(parameters)
        self.model.to(self.device)
        self.tokenizer = BertReranker.get_tokenizer(parameters)

        print("The reranking model is loaded")

    def rerank(self, mentions, sentences):
        model = self.model
        tokenizer = self.tokenizer
        p = self.parameters
        device = self.device

        data, tensor_data = BertReranker._process_mentions_for_model(
            p["context_key"],
            mentions,
            tokenizer,
            p["max_seq_length"],
            p["top_k"],
            p["silent"],
            sentences=sentences,
        )

        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=p["evaluation_batch_size"]
        )

        softmax = torch.nn.Softmax(dim=1)

        for input_ids, input_mask, segment_ids, mention_ids, entity_mask in tqdm(
            dataloader, desc="Inferring"
        ):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            mention_ids = mention_ids.numpy()
            entity_mask = entity_mask.to(device)

            with torch.no_grad():
                logits = self.model(
                    input_ids, segment_ids, input_mask, entity_mask=entity_mask
                )[0]
                probs = softmax(logits)

            logits = logits.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()

            predictions = np.argmax(logits, axis=1)

            for idx, mention_idx in enumerate(mention_ids):
                pred = predictions[idx].item()
                mentions[mention_idx]["predicted_candidate_idx"] = pred
                mentions[mention_idx]["prob_assigned_to_candidate"] = probs[idx][
                    pred
                ].item()

        return mentions

    def get_scheduler_and_optimizer(self, parameters, train_tensor_data, logger):
        model = self.model

        num_train_optimization_steps = (
            int(
                len(train_tensor_data)
                / parameters["train_batch_size"]
                / parameters["gradient_accumulation_steps"]
            )
            * parameters["num_train_epochs"]
        )
        num_warmup_steps = int(
            num_train_optimization_steps * parameters["warmup_proportion"]
        )
        param_optimizer = list(model.named_parameters())

        param_optimizer = [n for n in param_optimizer]

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=parameters["learning_rate"],
            correct_bias=False,
        )

        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_train_optimization_steps,
        )

        logger.info("  Num optimization steps = %d", num_train_optimization_steps)
        logger.info("  Num warmup steps = %d", num_warmup_steps)
        return optimizer, scheduler

    @staticmethod
    def get_model(parameters):
        model = BertForReranking.from_pretrained(
            parameters["path_to_model"],
            num_labels=parameters["top_k"],
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "local"),
        )

        if parameters["dataparallel_bert"]:
            model.bert = torch.nn.DataParallel(model.bert)
            print("Data parallel Bert")

        return model

    @staticmethod
    def get_tokenizer(parameters):
        tokenizer = BertTokenizer.from_pretrained(
            parameters["path_to_model"], do_lower_case=parameters["lowercase_flag"]
        )
        return tokenizer

    @staticmethod
    def _get_candidate_representation(
        context_tokens, candidate_desc, tokenizer, max_seq_length, max_sub_seq_length
    ):
        """Tokenizes and truncates description; combines it with the tokenized context and generates one input sample for bert"""
        candidate_desc_tokens = tokenizer.tokenize(candidate_desc)
        candidate_desc_tokens = candidate_desc_tokens[:max_sub_seq_length]

        tokens = (
            ["[CLS]"] + context_tokens + ["[SEP]"] + candidate_desc_tokens + ["[SEP]"]
        )
        segment_ids = [0] * (len(context_tokens) + 2) + [1] * (
            len(candidate_desc_tokens) + 1
        )
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }

    @staticmethod
    def _get_mention_context_end2end(mention, sentences):
        """Given a mention and a list of sentences that follow the blink conventions, it returns a left and right context for the mention"""
        sent_idx = mention["sent_idx"]

        prev_sent = sentences[sent_idx - 1] if sent_idx > 0 else ""
        next_sent = sentences[sent_idx + 1] if sent_idx + 1 < len(sentences) else ""
        prev_sent = sentences[sent_idx - 1] if False else ""
        next_sent = sentences[sent_idx + 1] if False else ""
        sent = sentences[sent_idx]

        curr_sent_prev = sent[: mention["start_pos"]].strip()
        curr_sent_next = sent[mention["end_pos"] :].strip()

        left_context = "{} {}".format(prev_sent, curr_sent_prev).strip()
        right_context = "{} {}".format(curr_sent_next, next_sent).strip()

        return (left_context, right_context)

    @staticmethod
    def _select_field(samples, field):
        """Helper function that returns a list of lists, each of which contains the information for all candidates for each sample"""
        return [
            [cand[field] for cand in sample["candidate_features"]] for sample in samples
        ]

    @staticmethod
    def _get_context_token_representation(
        context_key,
        sample,
        tokenizer,
        max_sub_seq_length,
        start_token,
        end_token,
        mention_text_key="text",
        tagged=True,
    ):
        """Tags the mention, trims the context and concatenates everything to form the context representation"""
        mention_tokens = (
            [start_token] + tokenizer.tokenize(sample[mention_text_key]) + [end_token]
        )

        max_sub_seq_length = (max_sub_seq_length - len(mention_tokens)) // 2

        context_left, context_right = sample[context_key]
        context_left = tokenizer.tokenize(context_left)
        context_right = tokenizer.tokenize(context_right)
        if len(context_left) > max_sub_seq_length:
            context_left = context_left[-max_sub_seq_length:]
        if len(context_right) > max_sub_seq_length:
            context_right = context_right[:max_sub_seq_length]

        context_tokens = context_left + mention_tokens + context_right

        return context_tokens

    @staticmethod
    def _process_mentions_for_model(
        context_key,
        mentions,
        tokenizer,
        max_seq_length,
        top_k,
        silent,
        start_token="[unused0]",
        end_token="[unused1]",
        debug=False,
        tagged=True,
        sentences=None,
        candidates_key="candidates",
        gold_key="gold_pos",
        logger=None,
    ):
        processed_mentions = []

        if debug:
            mentions = mentions[:200]

        max_sub_seq_length = (max_seq_length - 3) // 2

        if silent:
            iter_ = mentions
        else:
            iter_ = tqdm(mentions)

        for idx, mention in enumerate(iter_):
            # if sentences is not none that means that we are processing end2end data for inference
            if sentences is not None:
                mention[context_key] = BertReranker._get_mention_context_end2end(
                    mention, sentences
                )

            context_tokens = BertReranker._get_context_token_representation(
                context_key,
                mention,
                tokenizer,
                max_sub_seq_length,
                start_token,
                end_token,
            )

            candidates = mention[candidates_key]
            candidate_features = []

            for candidate in candidates[:top_k]:
                candidate_desc = " ".join(candidate["sentences"])

                candidate_obj = BertReranker._get_candidate_representation(
                    context_tokens,
                    candidate_desc,
                    tokenizer,
                    max_seq_length,
                    max_sub_seq_length,
                )
                candidate_features.append(candidate_obj)

            entity_mask = [1] * len(candidate_features) + [0] * (
                top_k - len(candidate_features)
            )

            if len(candidates) < top_k:
                candidate_desc = ""
                padding_candidate_obj = BertReranker._get_candidate_representation(
                    context_tokens,
                    candidate_desc,
                    tokenizer,
                    max_seq_length,
                    max_sub_seq_length,
                )
                for _ in range(top_k - len(candidates)):
                    candidate_features.append(padding_candidate_obj)

            assert len(candidate_features) == top_k
            assert len(entity_mask) == top_k

            if sentences is not None:
                processed_mentions.append(
                    {
                        "candidate_features": candidate_features,
                        "mention_idx": idx,
                        "entity_mask": entity_mask,
                    }
                )
            else:
                label = mention[gold_key] - 1
                processed_mentions.append(
                    {
                        "candidate_features": candidate_features,
                        "label": label,
                        "entity_mask": entity_mask,
                    }
                )

        all_input_ids = torch.tensor(
            BertReranker._select_field(processed_mentions, "input_ids"),
            dtype=torch.long,
        )
        all_input_mask = torch.tensor(
            BertReranker._select_field(processed_mentions, "input_mask"),
            dtype=torch.long,
        )
        all_segment_ids = torch.tensor(
            BertReranker._select_field(processed_mentions, "segment_ids"),
            dtype=torch.long,
        )
        all_entity_masks = torch.tensor(
            [s["entity_mask"] for s in processed_mentions], dtype=torch.float
        )

        data = {
            "all_input_ids": all_input_ids,
            "all_input_mask": all_input_mask,
            "all_segment_ids": all_segment_ids,
            "all_entity_masks": all_entity_masks,
        }

        if sentences is not None:
            all_mention_indices = torch.tensor(
                [s["mention_idx"] for s in processed_mentions], dtype=torch.long
            )
            data["all_mention_indices"] = all_mention_indices
            tensor_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_mention_indices,
                all_entity_masks,
            )
        else:
            all_label = torch.tensor(
                [s["label"] for s in processed_mentions], dtype=torch.long
            )
            data["all_label"] = all_label
            tensor_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_label,
                all_entity_masks,
            )

        if logger != None:
            logger.info("all_input_ids shape: {}".format(all_input_ids.shape))
            logger.info("all_input_mask shape: {}".format(all_input_mask.shape))
            logger.info("all_segment_ids shape: {}".format(all_segment_ids.shape))
            logger.info("all_entity_masks shape: {}".format(all_entity_masks.shape))
            if sentences is not None:
                logger.info(
                    "all_mention_indices shape: {}".format(all_mention_indices.shape)
                )
            else:
                logger.info("all_label shape: {}".format(all_label.shape))

        return data, tensor_data
