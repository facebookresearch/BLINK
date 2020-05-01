# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from flair.models import SequenceTagger
from flair.data import Sentence


def get_model(parameters=None):
    return Flair(parameters)


class NER_model:
    def __init__(self, parameters=None):
        pass

    def predict(self, sents):
        """Sents: List of plain text consequtive sentences. 
        Returns a dictionary consisting of a list of sentences and a list of mentions, where for each mention AT LEAST (it may give additional information) the following information is given:
            sent_idx - the index of the sentence that contains the mention
            text - the textual span that we hypothesise that represents an entity
            start_pos - the character idx at which the textual mention starts 
            end_pos - the character idx at which the mention ends"""
        pass


class Flair(NER_model):
    def __init__(self, parameters=None):
        self.model = SequenceTagger.load("ner")

    def predict(self, sentences):
        mentions = []
        for sent_idx, sent in enumerate(sentences):
            sent = Sentence(sent, use_tokenizer=True)
            self.model.predict(sent)
            sent_mentions = sent.to_dict(tag_type="ner")["entities"]
            for mention in sent_mentions:
                mention["sent_idx"] = sent_idx
            mentions.extend(sent_mentions)
        return {"sentences": sentences, "mentions": mentions}

