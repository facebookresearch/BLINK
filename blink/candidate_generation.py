# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import os
import pysolr
import sys

import blink.candidate_retrieval.utils as utils


def get_model(params):
    return BM45_Candidate_Generator(params)


class Candidate_Generator:
    def __init__(self, parameters=None):
        pass

    def get_candidates(self, mention_data):
        """Given the mentions from the named entity recognition model, generates candidates for each mention and adds them as an additional field to the mention dictionary"""
        pass


class BM45_Candidate_Generator(Candidate_Generator):
    ESCAPE_CHARS_RE = re.compile(r'(?<!\\)(?P<char>[&|+\-!(){}[\]\/^"~*?:])')

    def __init__(self, params):
        self.solr_address = params["solr_address"]
        self.raw_solr_fields = params["raw_solr_fields"]
        self.solr = pysolr.Solr(self.solr_address, always_commit=True, timeout=100)
        self.rows = params["rows"]
        self.query = params["query"]
        self.keys = [k.strip() for k in params["keys"].split(",")]
        self.c = 0
        self.query_arguments = {
            "fl": "* score",
            "rows": self.rows,
            "defType": "edismax",
        }

        if params["boosting"] is not None:
            self.query_arguments["bf"] = params["boosting"]

    def _filter_result(self, cand, detailed=True):
        wikidata_id = cand.get("wikidata_id", None)
        res = {
            "wikidata_id": wikidata_id,
            "wikipedia_id": cand["id"],
            "wikipedia_title": cand["title"],
        }

        if detailed:
            res["aliases"] = cand.get("aliases", None)
            sents = []

            for k in range(0, 10):
                key = "sent_desc_{}".format(k + 1)
                sents.append(cand.get(key, ""))

            res["sentences"] = sents

        return res

    def get_candidates(self, mention_data):
        solr = self.solr

        # Build query
        keys = self.keys
        query = self.query
        if not self.raw_solr_fields:
            query = query.format(
                *[
                    BM45_Candidate_Generator.solr_escape(mention_data[key])
                    if key in mention_data
                    else utils.get_sent_context(mention_data, key)
                    for key in keys
                ]
            )
        else:
            query = query.format(
                *[
                    mention_data[key]
                    if key in mention_data
                    else utils.get_sent_context(mention_data, key)
                    for key in keys
                ]
            )

        try:
            results = solr.search(query, **self.query_arguments)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("\nException:", exc_type, "- line", exc_tb.tb_lineno)
            print(repr(e))

            c = self.c
            if c < 10:
                print(
                    "Exception with: \naddress: {} \nquery: {} \nmention_data: {} \n".format(
                        self.solr_address, query, str(mention_data)
                    )
                )
            self.c = c + 1

            return []

        # Filter the data in the retrieved objects, while ignoring the ones without a wikidata_id (only a very small fraction in the dataset; they are noise)
        filtered_results = [
            self._filter_result(cand) for cand in results.docs if "wikidata_id" in cand
        ]
        return filtered_results

    @staticmethod
    def process_mentions_for_candidate_generator(sentences, mentions):
        for m in mentions:
            m["context"] = sentences[m["sent_idx"]]
        return mentions

    @staticmethod
    def solr_escape(string):
        if (string == "OR") or (string == "AND"):
            return string.lower()

        interior = r"\s+(OR|AND)\s+"
        start = r"^(OR|AND) "
        end = r" (OR|AND)$"

        string = re.sub(interior, lambda x: x.group(0).lower(), string)
        string = re.sub(start, lambda x: x.group(0).lower(), string)
        string = re.sub(end, lambda x: x.group(0).lower(), string)

        return BM45_Candidate_Generator.ESCAPE_CHARS_RE.sub(r"\\\g<char>", string)

