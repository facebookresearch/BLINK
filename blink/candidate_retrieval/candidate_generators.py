# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pysolr
import sys
import utils


def mention_data_summary(mention):
    return (mention["mention"], mention["query_truncated_25_context"])


class Simple_Candidate_Generator:
    def __init__(self, params):
        self.collection_name = params["collection_name"]
        self.solr_address = params["solr_address"]
        self.solr = pysolr.Solr(
            "{}/solr/{}".format(self.solr_address, self.collection_name),
            always_commit=True,
            timeout=100,
        )
        self.rows = params["rows"]
        self.query_data = params["query_data"]
        self.c = 0
        self.query_arguments = {
            "fl": "* score",
            "rows": self.rows,
            "defType": "edismax",
        }

        if params["boosting"] is not None:
            self.query_arguments["bf"] = params["boosting"]

    def _filter_result(self, cand):
        wikidata_id = cand.get("wikidata_id", None)
        res = {
            "wikidata_id": wikidata_id,
            "wikipedia_id": cand["id"],
            "wikipedia_title": cand["title"],
        }

        res["aliases"] = cand.get("aliases", None)
        sents = []

        for k in range(0, 10):
            key = "sent_desc_{}".format(k + 1)
            sents.append(cand.get(key, ""))

        res["sentences"] = sents

        res["num_incoming_links"] = cand.get("num_incoming_links", 0)
        res["score"] = cand["score"]

        return res

    def get_candidates(
        self,
        mention_data,
        verbose=False,
        print_number_of_docs_retrieved=False,
        print_query_flag=False,
    ):
        solr = self.solr
        query_data = self.query_data

        # Build query
        keys = query_data["keys"]
        query = query_data["string"]
        query = query.format(
            *[
                mention_data[key]
                if key in mention_data
                else utils.get_sent_context(mention_data, key)
                for key in keys
            ]
        )

        if print_query_flag:
            print("Query: {}".format(query))

        try:
            results = solr.search(query, **self.query_arguments)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("\nException:", exc_type, "- line", exc_tb.tb_lineno)
            print(repr(e))

            c = self.c
            c += 1
            if c < 10:
                print(
                    "Exception with: \ncollection_name: {} \nquery: {} \nmention_data: {} \ndataset_name: {}\nquery_args: {}\n".format(
                        self.collection_name,
                        query,
                        mention_data_summary(mention_data),
                        mention_data["dataset_name"],
                        str(self.query_arguments),
                    )
                )

            return []

        if print_number_of_docs_retrieved:
            print("Retrieved {0} result(s).".format(len(results)))
        # Return the full retrieved objects (debuging purposes)
        if verbose:
            return results

        # Filter the data in the retrieved objects, while ignoring the ones without a wikidata_id (only a very small fraction in the dataset; they are noise)
        filtered_results = [
            self._filter_result(cand) for cand in results.docs if "wikidata_id" in cand
        ]
        return filtered_results


class Pregenerated_Candidates_Data_Fetcher:
    def __init__(self, parameters):
        solr_address = "http://localhost:8983/solr/{}".format(
            parameters["collection_name"]
        )

        query_arguments = {"fl": "* score", "rows": 1, "defType": "edismax"}

        query_arguments["bf"] = "log(sum(num_incoming_links,1))"

        self.solr = pysolr.Solr(solr_address, always_commit=True, timeout=100)
        self.query_arguments = query_arguments

    def get_candidates_data(self, candidates_wikidata_ids):
        candidates_rich = []

        for candidate in candidates_wikidata_ids:
            candidate_data = self.get_candidate_data_for_wikidata_id(candidate[0])

            if candidate_data != None:
                candidate_data["p_e_m_score"] = candidate[2]
                candidates_rich.append(candidate_data)

        return candidates_rich

    @staticmethod
    def filter_result(cand, detailed=True):
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

    def get_candidate_data_for_wikidata_id(self, wikidata_id):
        results = self.solr.search(
            "wikidata_id:{}".format(wikidata_id), **self.query_arguments
        )

        if len(results) == 0:
            return None

        filtered_results = [
            Pregenerated_Candidates_Data_Fetcher.filter_result(cand)
            for cand in results.docs
            if "wikidata_id" in cand
        ]

        return filtered_results[0]
