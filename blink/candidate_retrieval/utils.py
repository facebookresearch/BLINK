# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import pickle
import subprocess
import blink.candidate_retrieval.dataset as D

import re
import os

ESCAPE_CHARS_RE = re.compile(r'(?<!\\)(?P<char>[&|+\-!(){}[\]\/^"~*?:])')


def solr_escape(string):
    if (string == "OR") or (string == "AND"):
        return string.lower()

    interior = r"\s+(OR|AND)\s+"
    start = r"^(OR|AND) "
    end = r" (OR|AND)$"

    string = re.sub(interior, lambda x: x.group(0).lower(), string)
    string = re.sub(start, lambda x: x.group(0).lower(), string)
    string = re.sub(end, lambda x: x.group(0).lower(), string)

    return ESCAPE_CHARS_RE.sub(r"\\\g<char>", string)


linktitle2id = None


def get_wikidata_id_from_link_name(link):
    global linktitle2id

    if linktitle2id is None:
        path_to_file = "data/KB_data/linktitle2wikidataid.p"
        if os.path.isfile(path_to_file):
            linktitle2id = pickle.load(open(path_to_file, "rb"))
        else:
            subprocess.call(
                "./blink/candidate_retrieval/scripts/generate_wiki2wikidata_mapping.sh"
            )
            linktitle2id = pickle.load(open(path_to_file, "rb"))

    return linktitle2id.get(link, None)


def get_datasets(get_test_dataset=False, get_pregenereted_candidates_wikidata_id=False):
    train_and_benchmarking_data_dir = "data/train_and_benchmark_data"
    datadir = os.path.join(
        train_and_benchmarking_data_dir, "generated/test_train_data/"
    )
    conll_path = os.path.join(
        train_and_benchmarking_data_dir, "basic_data/test_datasets/"
    )
    person_path = os.path.join(
        train_and_benchmarking_data_dir, "basic_data/p_e_m_data/persons.txt"
    )
    p_e_m_path = os.path.join(train_and_benchmarking_data_dir, "basic_data/p_e_m_data/")

    added_params = {
        "generate_cands": False,
        "generate_ments_and_cands": False,
        "candidate_generator_type": "p_e_m",
        "p_e_m_data_path": p_e_m_path,
    }
    conll = D.CoNLLDataset(datadir, person_path, conll_path, added_params)

    dev_datasets = [
        ("aida-A", conll.testA),
        ("aida-B", conll.testB),
        ("msnbc", conll.msnbc),
        ("aquaint", conll.aquaint),
        ("ace2004", conll.ace2004),
        ("clueweb", conll.clueweb),
        ("wikipedia", conll.wikipedia),
    ]

    if get_test_dataset:
        dev_datasets.append(("aida-train", conll.train))

    not_found = []
    total = 0
    for ds_name, dataset in dev_datasets:
        print("Processing dataset:", ds_name)
        for doc_name, content in dataset.items():
            for m in content:
                total += 1
                link = m["gold"][0]
                wikidata_id = get_wikidata_id_from_link_name(link)

                if wikidata_id is None:
                    not_found.append(m)

                m["gold_wikidata_id"] = wikidata_id

                if get_pregenereted_candidates_wikidata_id:
                    cands = []
                    for candidate in m["candidates"]:
                        link, prob = candidate
                        wikidata_id = get_wikidata_id_from_link_name(link)
                        cands.append((wikidata_id, link, prob))
                    m["candidates_wikidata_ids"] = cands

    print("Number of entities:", total)
    print(
        "Wikidata ID not found for:",
        len(not_found),
        "({:.3f} %)".format(len(not_found) * 1.0 / total),
    )

    return dev_datasets


def get_sent_context(mention, key, solr_escaped=True):
    if not solr_escaped:
        mention_data_key = "sent_context_orig"
    else:
        mention_data_key = "sent_context"

    if key.endswith("next"):
        if key.endswith("prev_next"):
            res = "{} {} {}".format(
                ""
                if mention[mention_data_key][0] is None
                else mention[mention_data_key][0],
                mention[mention_data_key][1],
                ""
                if mention[mention_data_key][2] is None
                else mention[mention_data_key][2],
            )
        else:
            res = "{} {}".format(
                mention[mention_data_key][1],
                ""
                if mention[mention_data_key][2] is None
                else mention[mention_data_key][2],
            )
    elif key.endswith("prev"):
        res = "{} {}".format(
            ""
            if mention[mention_data_key][0] is None
            else mention[mention_data_key][0],
            mention[mention_data_key][1],
        )
    else:
        res = mention[mention_data_key][1]

    return res.strip()


def get_list_of_mentions(dev_datasets):
    mentions = []

    total_invalid = 0
    total_valid = 0

    for ds_name, dataset in dev_datasets:
        invalid = 0
        valid = 0

        print("Processing dataset:", ds_name)
        for doc_name, content in dataset.items():
            sentences = content[0]["conll_doc"]["sentences"]
            for m in content:
                gold_wikidata_id = m["gold_wikidata_id"]
                left_context, right_context = m["context"]

                m["mention_orig"] = m["mention"]
                m["mention"] = solr_escape(m["mention"])

                if left_context != "EMPTYCTXT":
                    left_context_orig = left_context
                    left_context = solr_escape(left_context)
                else:
                    left_context = ""

                if right_context != "EMPTYCTXT":
                    right_context_orig = right_context
                    right_context = solr_escape(right_context)
                else:
                    right_context = ""

                m["left_context_orig"] = left_context_orig
                m["right_context_orig"] = right_context_orig

                m["query_context"] = "{} {} {}".format(
                    left_context, m["mention"], right_context
                ).strip()
                m["query_context_orig"] = "{} {} {}".format(
                    left_context_orig, m["mention_orig"], right_context_orig
                ).strip()

                truncated_left_context = " ".join(left_context.split(" ")[-25:])
                truncated_right_context = " ".join(right_context.split(" ")[:25])
                m["query_truncated_25_context"] = "{} {} {}".format(
                    truncated_left_context, m["mention"], truncated_right_context
                ).strip()

                truncated_left_context = " ".join(left_context.split(" ")[-10:])
                truncated_right_context = " ".join(right_context.split(" ")[:10])
                m["query_truncated_10_context"] = "{} {} {}".format(
                    truncated_left_context, m["mention"], truncated_right_context
                ).strip()

                m["dataset_name"] = ds_name
                m["doc_name"] = doc_name

                sent_id, start, end = (
                    m["conll_m"]["sent_id"],
                    m["conll_m"]["start"],
                    m["conll_m"]["end"],
                )
                prev_sent_id = sent_id - 1
                next_sent_id = sent_id + 1

                sent_orig = " ".join(sentences[sent_id]).strip()
                m["left_query_sent_context_orig"] = " ".join(sentences[sent_id][:start])
                m["right_query_sent_context_orig"] = " ".join(sentences[sent_id][end:])
                sent = solr_escape(sent_orig)

                # try:
                #     context_parts_lower = '{} {} {}'.format(m['left_query_sent_context_orig'], m['mention_orig'], m['right_query_sent_context_orig']).strip().lower()
                #     context_orig_lower = sent_orig.lower()
                #     assert(context_parts_lower == context_orig_lower)
                # except:
                #     print(context_parts_lower)
                #     print(context_orig_lower)
                #     input("")

                if prev_sent_id > 0:
                    prev_sent_orig = " ".join(sentences[prev_sent_id])
                    prev_sent = solr_escape(prev_sent_orig)
                else:
                    prev_sent_orig = None
                    prev_sent = None

                if next_sent_id < len(sentences):
                    next_sent_orig = " ".join(sentences[next_sent_id])
                    next_sent = solr_escape(next_sent_orig)
                else:
                    next_sent_orig = None
                    next_sent = None

                m["sent_context"] = (prev_sent, sent, next_sent)
                m["sent_context_orig"] = (prev_sent_orig, sent_orig, next_sent_orig)
                # m['sent_context_prev'] = get_sent_context(m, 'sent_context_prev')
                # m['sent_context_next'] = get_sent_context(m, 'sent_context_next')
                # m['sent_context_prev_next'] = get_sent_context(m, 'sent_context_prev_next')
                # m['sent_context_curr'] = get_sent_context(m, 'sent_context_curr')

                if gold_wikidata_id is None:
                    invalid += 1
                    continue

                mentions.append(m)
                valid += 1

        print("Invalid: ", invalid)
        print("Valid: ", valid)

        total_invalid += invalid
        total_valid += valid

    return mentions


def write_candidate_generation_results_for_a_run_to_file(run, results_dump_file_path):
    txt_file_path = "{}.txt".format(results_dump_file_path)

    with open(txt_file_path, "a+") as file:
        id_ = "Q: `{}` === K: `{}` === ID: `{}`".format(
            run[0]["query"], run[0]["keys"], run[0]["dump_file_id"]
        )
        res = " --- ".join(
            ["{} - {:.2f}".format(key, run[1][key]) for key in sorted(run[1].keys())]
        )
        file.write("{} === {}\n".format(res, id_))


def write_candidate_generation_execution_time_to_file(
    results_dump_file_path, execution_time
):
    txt_file_path = "{}.txt".format(results_dump_file_path)

    with open(txt_file_path, "a+") as file:
        file.write("The execution took: {} minutes".format(execution_time))


def write_candidate_generation_results_to_file(
    runs, results_dump_file_path, execution_time=None
):
    runs.sort(key=lambda run: -run[1]["overall"])

    for run in runs:
        write_candidate_generation_results_for_a_run_to_file(
            run, results_dump_file_path
        )

    if execution_time is not None:
        write_candidate_generation_execution_time_to_file(
            results_dump_file_path, execution_time
        )

