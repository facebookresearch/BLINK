# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import emoji


def get_model(parameters):
    return Wikimedia_Data_Fetcher(parameters["path_to_candidate_data_dict"])


class Wikimedia_Data_Fetcher:
    def __init__(self, path_to_data):
        self.data = pickle.load(open(path_to_data, "rb"))

    def get_data_for_entity(self, entity_data):
        """Given an entity data dictionary that contains some linking data (ex. title or ID), additional information (ex. description, aliases etc.) is added to the given entity dictionary"""
        data = self.data
        title = entity_data["wikipedia_title"]

        if "wikidata_info" in data[title]:
            if ("aliases" in data[title]["wikidata_info"]) and (
                data[title]["wikidata_info"]["aliases"]
            ) is not None:
                aliases = [
                    alias
                    for alias in data[title]["wikidata_info"]["aliases"]
                    if alias not in emoji.UNICODE_EMOJI
                ]
            else:
                aliases = None
        else:
            aliases = None

        entity_data["aliases"] = aliases

        sents = []

        for k in range(0, 10):
            key = "sent_desc_{}".format(k + 1)
            sents.append(data[title].get(key, ""))

        entity_data["sentences"] = sents

        return entity_data
