# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import io
import json
import os
import pickle

from segtok.segmenter import split_multi

##### Reading helpers #####
def read_sentences_from_file(path_to_file, one_sentence_per_line=True):
    lines = []
    with io.open(path_to_file, mode="r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line != "":
                lines.append(line.strip())

    if one_sentence_per_line:
        sentences = lines
    else:
        text = " ".join(lines)
        sentences = list(split_multi(text))
        sentences = [sentence for sentence in sentences if sentence != ""]

    return sentences


##### Printing / writing  helpers #####
def get_candidate_summary(candidate):
    wikipedia_id = candidate["wikipedia_id"]
    wikidata_id = candidate["wikidata_id"]
    wikipedia_title = candidate["wikipedia_title"]

    return "{}, {}, {}".format(wikipedia_id, wikidata_id, wikipedia_title)


def present_sentence_mentions(sentence, mentions, output_file):
    if output_file != None:
        f = io.open(output_file, mode="a", encoding="utf-8")
        output = lambda s: f.write("{}\n".format(s))
    else:
        output = lambda s: print(s)
    output("Sentence: {}".format(sentence))

    mention_entity_pairs = []
    for mention in mentions:
        candidates = mention["candidates"]
        # prediction = mention.get('predicted_candidate_idx', 0)
        prediction = mention["predicted_candidate_idx"]

        if prediction < len(candidates):
            # print(type(mention['prob_assigned_to_candidate']))
            # print(mention['prob_assigned_to_candidate'])
            mention_rep = "{} ({}, {}) - {} (conf. {:.5f})".format(
                mention["text"],
                mention["start_pos"],
                mention["end_pos"],
                get_candidate_summary(candidates[prediction]),
                mention["prob_assigned_to_candidate"],
            )
        else:
            mention_rep = "{} ({}, {}) - {}".format(
                mention["text"],
                mention["start_pos"],
                mention["end_pos"],
                "No candidate selected",
            )

        mention_entity_pairs.append(mention_rep)

    if len(mention_entity_pairs) != 0:
        output("Mention-Entity pairs: \n{}".format("\n".join(mention_entity_pairs)))
    else:
        output("No detected mentions")

    output("")


def sentence_mentions_pairs(sentences, mentions):
    mentions_per_sent = {}

    for m in mentions:
        sent_idx = int(m["sent_idx"])

        curr_ments = mentions_per_sent.get(sent_idx, [])
        curr_ments.append(m)

        mentions_per_sent[sent_idx] = curr_ments

    pairs = []

    for idx, sent in enumerate(sentences):
        pairs.append((sent, mentions_per_sent.get(idx, [])))

    return pairs


def present_annotated_sentences(sentences, mentions, output_file=None):
    pairs = sentence_mentions_pairs(sentences, mentions)

    for sent, ments in pairs:
        present_sentence_mentions(sent, ments, output_file)


def write_dicts_as_json_per_line(list_of_dicts, txt_file_path):
    with io.open(txt_file_path, mode="w", encoding="utf-8") as file:
        for idx, mention in enumerate(list_of_dicts):
            json_string = json.dumps(mention)
            file.write(json_string)

            if idx != (len(list_of_dicts) - 1):
                file.write("\n")


def get_mentions_txt_file_path(output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    file_name = "mentions.jsonl"
    path_to_file = os.path.join(output_folder_path, file_name)

    return path_to_file


def get_sentences_txt_file_path(output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    file_name = "sentences.jsonl"
    path_to_file = os.path.join(output_folder_path, file_name)

    return path_to_file


def get_end2end_pickle_output_file_path(output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    file_name = "mentions_and_sentences.pickle"
    path_to_file = os.path.join(output_folder_path, file_name)

    return path_to_file


def write_end2end_pickle_output(sentences, mentions, output_file_id):
    obj = {"sentences": sentences, "mentions": mentions}
    with open(get_end2end_pickle_output_file_path(output_file_id), "wb") as file:
        pickle.dump(obj, file)


def get_end2end_pretty_output_file_path(output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    file_name = "pretty.txt"
    path_to_file = os.path.join(output_folder_path, file_name)
    return path_to_file
