# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import pickle
import os
import time
import numpy as np

"""
This script is adapted from https://github.com/lephong/mulrel-nel
"""


def read_csv_file(path, added_params):
    data = {}
    info = True
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            comps = line.strip().split("\t")
            doc_name = comps[0] + " " + comps[1]
            mention = comps[2]
            lctx = comps[3]
            rctx = comps[4]

            if comps[6] != "EMPTYCAND":
                cands = [c.split(",") for c in comps[6:-2]]
                cands = [
                    (",".join(c[2:]).replace('"', "%22").replace(" ", "_"), float(c[1]))
                    for c in cands
                ]
            else:
                cands = []

            gold = comps[-1].split(",")
            if gold[0] == "-1":
                gold = (
                    ",".join(gold[2:]).replace('"', "%22").replace(" ", "_"),
                    1e-5,
                    -1,
                )
            else:
                gold = (
                    ",".join(gold[3:]).replace('"', "%22").replace(" ", "_"),
                    1e-5,
                    -1,
                )

            if added_params["generate_cands"]:
                if info:
                    print("Generating candidates")
                    info = False
                cands = added_params["cand_generator"].process(mention)

            if doc_name not in data:
                data[doc_name] = []

            data[doc_name].append(
                {
                    "mention": mention,
                    "context": (lctx, rctx),
                    "candidates": cands,
                    "gold": gold,
                }
            )
    return data


### Adds original textual data to pregenerated data
def read_conll_file(data, path):
    conll = {}
    with open(path, "r", encoding="utf8") as f:
        cur_sent = None
        cur_doc = None

        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                docname = line.split()[1][1:]
                conll[docname] = {"sentences": [], "mentions": []}
                cur_doc = conll[docname]
                cur_sent = []

            else:
                if line == "":
                    cur_doc["sentences"].append(cur_sent)
                    cur_sent = []

                else:
                    comps = line.split("\t")
                    tok = comps[0]
                    cur_sent.append(tok)

                    if len(comps) >= 6:
                        bi = comps[1]
                        wikilink = comps[4]
                        if bi == "I":
                            cur_doc["mentions"][-1]["end"] += 1
                        else:
                            new_ment = {
                                "sent_id": len(cur_doc["sentences"]),
                                "start": len(cur_sent) - 1,
                                "end": len(cur_sent),
                                "wikilink": wikilink,
                            }
                            cur_doc["mentions"].append(new_ment)

    # merge with data
    rmpunc = re.compile("[\W]+")
    for doc_name, content in data.items():
        conll_doc = conll[doc_name.split()[0]]
        content[0]["conll_doc"] = conll_doc

        cur_conll_m_id = 0
        for m in content:
            mention = m["mention"]
            gold = m["gold"]

            while True:
                cur_conll_m = conll_doc["mentions"][cur_conll_m_id]
                cur_conll_mention = " ".join(
                    conll_doc["sentences"][cur_conll_m["sent_id"]][
                        cur_conll_m["start"] : cur_conll_m["end"]
                    ]
                )
                if rmpunc.sub("", cur_conll_mention.lower()) == rmpunc.sub(
                    "", mention.lower()
                ):
                    m["conll_m"] = cur_conll_m
                    cur_conll_m_id += 1
                    break
                else:
                    cur_conll_m_id += 1

    return data


##### Check whether an entity is a person and if the doc contains other references with a more descriptive name for the person
##### (ex. John vs John Snow vs John Snow Stark). Then processes the candidate lists for all of the mentions that fit this description.


def load_person_names(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data.append(line.strip().replace(" ", "_"))
    return set(data)


def find_coref(ment, mentlist, person_names):
    cur_m = ment["mention"].lower()
    coref = []
    for m in mentlist:
        if len(m["candidates"]) == 0 or m["candidates"][0][0] not in person_names:
            continue

        mention = m["mention"].lower()
        start_pos = mention.find(cur_m)
        if start_pos == -1 or mention == cur_m:
            continue

        end_pos = start_pos + len(cur_m) - 1
        if (start_pos == 0 or mention[start_pos - 1] == " ") and (
            end_pos == len(mention) - 1 or mention[end_pos + 1] == " "
        ):
            coref.append(m)

    return coref


def with_coref(dataset, person_names):
    for data_name, content in dataset.items():
        for cur_m in content:
            coref = find_coref(cur_m, content, person_names)
            if coref is not None and len(coref) > 0:
                cur_cands = {}
                for m in coref:
                    for c, p in m["candidates"]:
                        cur_cands[c] = cur_cands.get(c, 0) + p
                for c in cur_cands.keys():
                    cur_cands[c] /= len(coref)
                cur_m["candidates"] = sorted(
                    list(cur_cands.items()), key=lambda x: x[1]
                )[::-1]


######


def eval(testset, system_pred, nel=False):
    gold = []
    pred = []

    for doc_name, content in testset.items():
        gold += [c["gold"][0] for c in content]  # the gold named entity
        pred += [
            c["pred"][0] for c in system_pred[doc_name]
        ]  # the predicted named entity

    true_pos = 0
    for g, p in zip(gold, pred):
        if g == p and p != "NIL":
            true_pos += 1

    if nel:
        NIL_preds = len([p for p in pred if p == "NIL"])
        total_discovered_mentions = 0
        for doc_name, content in testset.items():
            total_discovered_mentions += np.sum(
                len(ment) for ment in content[0]["ments_per_sent_flair"]
            )

        precision = true_pos / (total_discovered_mentions - NIL_preds)
    else:
        precision = true_pos / len([p for p in pred if p != "NIL"])

    recall = true_pos / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_candidate_generator(added_params):
    if added_params["candidate_generator_type"] == "p_e_m":
        if "p_e_m_data_path" in added_params:
            return FetchCandidateEntities(added_params["p_e_m_data_path"])
        else:
            return FetchCandidateEntities()
    else:
        pass


class CoNLLDataset:
    """
    reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, path, person_path, conll_path, added_params):
        if added_params["generate_ments_and_cands"]:
            added_params["generate_cands"] = False

        if added_params["generate_cands"] or added_params["generate_ments_and_cands"]:
            added_params["cand_generator"] = get_candidate_generator(added_params)

        print(added_params)

        print("load csv")
        self.train = read_csv_file(path + "/aida_train.csv", added_params)
        self.testA = read_csv_file(path + "/aida_testA.csv", added_params)
        self.testB = read_csv_file(path + "/aida_testB.csv", added_params)
        self.ace2004 = read_csv_file(path + "/wned-ace2004.csv", added_params)
        self.aquaint = read_csv_file(path + "/wned-aquaint.csv", added_params)
        self.clueweb = read_csv_file(path + "/wned-clueweb.csv", added_params)
        self.msnbc = read_csv_file(path + "/wned-msnbc.csv", added_params)
        self.wikipedia = read_csv_file(path + "/wned-wikipedia.csv", added_params)
        self.wikipedia.pop("Jiří_Třanovský Jiří_Třanovský", None)

        print("process coref")
        person_names = load_person_names(person_path)
        with_coref(self.train, person_names)
        with_coref(self.testA, person_names)
        with_coref(self.testB, person_names)
        with_coref(self.ace2004, person_names)
        with_coref(self.aquaint, person_names)
        with_coref(self.clueweb, person_names)
        with_coref(self.msnbc, person_names)
        with_coref(self.wikipedia, person_names)

        print("load conll")
        read_conll_file(self.train, conll_path + "/AIDA/aida_train.txt")
        read_conll_file(self.testA, conll_path + "/AIDA/testa_testb_aggregate_original")
        read_conll_file(self.testB, conll_path + "/AIDA/testa_testb_aggregate_original")
        read_conll_file(
            self.ace2004, conll_path + "/wned-datasets/ace2004/ace2004.conll"
        )
        read_conll_file(
            self.aquaint, conll_path + "/wned-datasets/aquaint/aquaint.conll"
        )
        read_conll_file(self.msnbc, conll_path + "/wned-datasets/msnbc/msnbc.conll")
        read_conll_file(
            self.clueweb, conll_path + "/wned-datasets/clueweb/clueweb.conll"
        )
        read_conll_file(
            self.wikipedia, conll_path + "/wned-datasets/wikipedia/wikipedia.conll"
        )

        if added_params["generate_cands"]:
            print(
                "Number of candidates not present in p_e_m originally, but present when lowercased",
                len(added_params["cand_generator"].lower_org),
            )
            print(
                "Number of candidates not present in p_e_m originally, but present in p_e_m_lower when lowercased ",
                len(added_params["cand_generator"].lower_lower),
            )


class FetchCandidateEntities(object):
    """takes as input a string or a list of words and checks if it is inside p_e_m
    if yes it returns the candidate entities otherwise it returns None.
    it also checks if string.lower() inside p_e_m and if string.lower() inside p_e_m_low"""

    def __init__(self, p_e_m_data_path="data/basic_data/p_e_m_data/"):
        print("Reading p_e_m dictionaries")
        # return
        wall_start = time.time()
        self.lower_org = []
        self.lower_lower = []
        self.p_e_m = pickle.load(
            open(os.path.join(p_e_m_data_path, "p_e_m_dict.pickle"), "rb")
        )
        self.p_e_m_lower = pickle.load(
            open(os.path.join(p_e_m_data_path, "p_e_m_lower_dict.pickle"), "rb")
        )
        self.mention_total_freq = pickle.load(
            open(os.path.join(p_e_m_data_path, "mention_total_freq.pickle"), "rb")
        )
        print("The reading took:", (time.time() - wall_start) / 60, " minutes")

    def process(self, span):
        """span can be either a string or a list of words"""

        title = span.title()
        # 'obama 44th president of united states'.title() # 'Obama 44Th President Of United States'
        title_freq = (
            self.mention_total_freq[title] if title in self.mention_total_freq else 0
        )
        span_freq = (
            self.mention_total_freq[span] if span in self.mention_total_freq else 0
        )

        if title_freq == 0 and span_freq == 0:
            if span.lower() in self.p_e_m:
                self.lower_org.append(span)
                return self.p_e_m[span.lower()]
            elif span.lower() in self.p_e_m_lower:
                self.lower_lower.append(span)
                return self.p_e_m_lower[span.lower()]
            else:
                return []
        else:
            if span_freq > title_freq:
                return self.p_e_m[span]
            else:
                return self.p_e_m[title]
