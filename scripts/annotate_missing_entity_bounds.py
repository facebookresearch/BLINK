import pdb
import json
import os
import numpy

THRESHOLD = 0.0

all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_ent_id_to_ents = {line['kb_idx']: line['title'] for line in all_wiki_ents if 'kb_idx' in line}
print("Created wikipedia ID -> title map")


import difflib
from Levenshtein import ratio
import itertools
import string

ALPHABET = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.digits)

def get_closest_word_bounds(string, char_idx, is_beginning):
    """
    is_beginning: boolean variable stating whether this bound is a beginning or ending boundary
        * affects how we compute boundaries
        (beginning boundaries should be after whitespaces, end boundaries before)
        (for beginning boundaries, we check whether idx-1 is whitespace,
         for end boundaries, we check whether idx is whitespace)
    """
    # before is inclusive
    # after is non-inclusive
    closest_before_bound = char_idx
    closest_after_bound = char_idx
    # if current boundary is at a word boundary, return only the current boundary
    if (
        is_beginning and (char_idx == 0 or string[char_idx - 1] not in ALPHABET)
    ) or (
        not is_beginning and (char_idx == len(string) or string[char_idx] not in ALPHABET)
    ):
        return [char_idx]
    while closest_before_bound > 0 and ((
        is_beginning and string[closest_before_bound - 1] in ALPHABET
    ) or (
        not is_beginning and string[closest_before_bound] in ALPHABET
    )):
        closest_before_bound -= 1
    while closest_after_bound < len(string) and ((
        is_beginning and string[closest_after_bound - 1] in ALPHABET
    ) or (
        not is_beginning and string[closest_after_bound] in ALPHABET
    )):
        closest_after_bound += 1
    # otherwise, return the 2 candidates
    return [closest_before_bound, closest_after_bound]


# longest substring match
def matches(large_string, query_string):
    # matches = []
    s = difflib.SequenceMatcher(None, large_string, query_string)
    match = ''.join(large_string[i:i+n] for i, j, n in s.get_matching_blocks() if n)
    min_begin = min([i for i, j, n in s.get_matching_blocks() if n])
    max_end = max([i+n for i, j, n in s.get_matching_blocks() if n])
    # standardize to word boundaries
    # 4 combinations: a[b cd e]f -> [ab cd] ef / ab [cd ef] / ab [cd] ef / [ab cd ef]
    # a[b c]d -> [ab] cd / ab [cd] / [ab cd]
    # choose based on: 1. closest similarity, 2. (tiebreaker) smallest span w/ at least one word
    closest_to_beginning_word_bounds = get_closest_word_bounds(large_string, min_begin, is_beginning=True)
    closest_to_end_word_bounds = get_closest_word_bounds(large_string, max_end, is_beginning=False)
    # validate the 4 combinations
    combinations = [
        (x, y, ratio(large_string[x:y], query_string))
        for x in closest_to_beginning_word_bounds
        for y in closest_to_end_word_bounds
        if y > x
    ]
    best_combo = combinations[0]
    for i, c in enumerate(combinations):
        if i == 0:
            continue
        if c[2] > best_combo[2]:
            best_combo = c
    bounded_match = large_string[best_combo[0]:best_combo[1]]
    # if len(match) / float(len(query_string)) >= threshold:
    #     matches.append(match)
    return match, bounded_match, best_combo[0], best_combo[1]


def annotate_examples():
    webqsp_save_files = []
    graphqs_save_files = []
    webqsp_subsets = {}
    num_correct = 0
    num_pred = 0
    num_actual = 0
    # load webqsp
    unannotated_examples = {}
    # TODO UNCOMMENT!!!!!
    for subset in ["train", "dev", "test"]:
        with_classes = ""
        if subset == "test":
            with_classes = ".with_classes"
        fn = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.{}.entities{}.json".format(subset, with_classes)
        with open(fn) as f:
            webqsp_subset = json.load(f)
        webqsp_subsets[subset] = webqsp_subset
        unannotated_examples[subset] = {}
        for i in range(len(webqsp_subset)):
            for j in range(len(webqsp_subset[i]['entities'])):
                target_id = webqsp_subset[i]['entities'][j]
                # check if we can get the main entity using fuzzy matching
                if target_id in all_wiki_ent_id_to_ents:
                    raw_match_chars, bounded_match, min_start, max_end = matches(
                        webqsp_subset[i]['utterance'], all_wiki_ent_id_to_ents[target_id].lower(),
                    )
                else:
                    raw_match_chars = None
                    bounded_match = None
                    min_start = 0
                    max_end = 0
                if target_id == webqsp_subset[i]['main_entity']:
                    if target_id in all_wiki_ent_id_to_ents:
                        # matches = find_near_matches(all_wiki_ent_id_to_ents[target_id].lower(), webqsp_subset[i]['utterance'])
                        # maximum start and end
                        # pdb.set_trace()
                        # for match in matches:
                        if min_start == webqsp_subset[i]['main_entity_pos'][0] and max_end == webqsp_subset[i]['main_entity_pos'][1]:
                            # if webqsp_subset[i]['utterance'][min_start:max_end] == webqsp_subset[i]['main_entity_tokens']:
                            num_correct += 1
                        num_pred += 1
                        num_actual += 1
                if target_id != None and target_id != webqsp_subset[i]['main_entity']:  # and match is not None:
                    question_hypothesis = webqsp_subset[i]['utterance']
                    question_hypothesis = question_hypothesis[:max_end] + ']' + question_hypothesis[max_end:]
                    question_hypothesis = question_hypothesis[:min_start] + '[' + question_hypothesis[min_start:]
                    unannotated_examples[subset][i] = {
                        "target": all_wiki_ent_id_to_ents[target_id] if target_id in all_wiki_ent_id_to_ents else "entity not in our dataset",
                        "question_with_hypothesis": question_hypothesis,
                        "question": webqsp_subset[i]['utterance'],
                        "target_ID": target_id,
                        "target_list_idx": j,
                        "question_with_main": (
                            webqsp_subset[i]['utterance'][:webqsp_subset[i]['main_entity_pos'][0]] + "_" \
                            + webqsp_subset[i]["main_entity_tokens"] + "_" \
                            + webqsp_subset[i]['utterance'][webqsp_subset[i]['main_entity_pos'][1]:]
                        ),
                        "main_entity": webqsp_subset[i]['main_entity_text'],
                        "qid": webqsp_subset[i]['question_id'],
                    }
        print(len(unannotated_examples[subset]))
        # save subset
        save_file = "to_annotate_mention_bound/predicted_bounds_{}_webqsp.json".format(subset)
        json.dump(unannotated_examples[subset], open(save_file, "w"), indent=2)
        webqsp_save_files.append(save_file)
    # graphqs
    graphqs_subsets = {}
    unannotated_examples = {}
    for subset in ["train", "test"]:
        with_classes = ""
        fn = "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{}.entities.json".format(subset)
        with open(fn) as f:
            graphqs_subset = json.load(f)
        graphqs_subsets[subset] = graphqs_subset
        unannotated_examples[subset] = {}
        for i in range(len(graphqs_subset)):
            for j in range(len(graphqs_subset[i]['entities'])):
                target_id = graphqs_subset[i]['entities'][j]
                target_ent = graphqs_subset[i]['entities_fb'][j].replace('_', ' ')
                # if target_id in all_wiki_ent_id_to_ents:
                #     target_ent = all_wiki_ent_id_to_ents[target_id].lower()
                # else:
                #     target_ent = graphqs_subset[i]['entities_fb'][j].replace('_', ' ')
                raw_match_chars, bounded_match, min_start, max_end = matches(graphqs_subset[i]['utterance'], target_ent)
                if target_id != None:  # and match is not None:
                    question_hypothesis = graphqs_subset[i]['utterance']
                    question_hypothesis = question_hypothesis[:max_end] + ']' + question_hypothesis[max_end:]
                    question_hypothesis = question_hypothesis[:min_start] + '[' + question_hypothesis[min_start:]
                    unannotated_examples[subset][i] = {
                        "target": all_wiki_ent_id_to_ents[target_id] if target_id in all_wiki_ent_id_to_ents else "entity not in our dataset",
                        "question_with_hypothesis": question_hypothesis,
                        "question": graphqs_subset[i]['utterance'],
                        "target (FB)": graphqs_subset[i]['entities_fb'][j].replace('_', ' '),
                        "target_ID": target_id,
                        "target_list_idx": j,
                        "qid": graphqs_subset[i]['question_id'],
                    }
        print(len(unannotated_examples[subset]))
        # save subset
        save_file = "to_annotate_mention_bound/predicted_bounds_{}_graphqs.json".format(subset)
        json.dump(unannotated_examples[subset], open(save_file, "w"), indent=2)
        graphqs_save_files.append(save_file)
    return webqsp_save_files, graphqs_save_files


# IF HYPOTHESIS CORRECT, LEAVE ALONE (don't need to go in and modify)
# IF ENTITY DOESN"T EXIST IN QUESTION, DELETE "QUESTION_HYPOTHESIS" KEY
# OTHERWISE (HYPOTHESIS INCORRECT), ANNOTATE

def filter_examples(fn, threshold):
    all_filtered_examples = {}
    with open(fn) as f:
        loaded_f = json.load(f)
        for q_idx in loaded_f:
            mention_start = loaded_f[q_idx]["question_with_hypothesis"].find("[")
            mention_end = loaded_f[q_idx]["question_with_hypothesis"].find("]")
            mention = loaded_f[q_idx]["question_with_hypothesis"][mention_start + 1:mention_end]
            if ratio(mention, loaded_f[q_idx]["target"].lower()) < threshold and loaded_f[q_idx]["target"] != "entity not in wikipedia":
                # only keep ones > threshold similarity ratio
                all_filtered_examples[q_idx] = loaded_f[q_idx]
        # save subset
        print("{} - {} = {} examples remaining".format(
            len(loaded_f),
            len(loaded_f) - len(all_filtered_examples),
            len(all_filtered_examples),
        ))
        save_file = "{}_filtered_out_{}.json".format(fn[:len(fn)-len(".json")], threshold)
        json.dump(all_filtered_examples, open(save_file, "w"), indent=2)
    return save_file


PARENT_DIR = "to_annotate_mention_bound"
divide_sets = {
    "predicted_bounds_train_webqsp_filtered_out_1.0.json": [66, 10],
    "predicted_bounds_dev_webqsp_filtered_out_1.0.json": [10, 7],
    "predicted_bounds_test_webqsp_filtered_out_1.0.json": [47, 10],
    "predicted_bounds_train_graphqs_filtered_out_1.0.json": [318, 8],
    "predicted_bounds_test_graphqs_filtered_out_1.0.json": [324, 8],
}

names = ["belinda", "sewon", "srini", "scott"]

def divide_datasets():
    save_dirs = []
    for divide_set_fn in divide_sets:
        with open(os.path.join(PARENT_DIR, divide_set_fn)) as f:
            loaded_f = json.load(f)
            print(len(loaded_f))
            loaded_f = [[k, v] for k, v in loaded_f.items()]
            for i in range(len(names)):
                save_dir = "to_annotate_mention_bound/annotation_set_{}".format(names[i])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_dirs.append(save_dir)
                save_file = os.path.join(save_dir, "{}.json".format(divide_set_fn[
                    len("predicted_bounds_"):len(divide_set_fn) - len("_filtered_out_1.0.json")
                ]))
                start = divide_sets[divide_set_fn][0] * i
                end = divide_sets[divide_set_fn][0] * (i + 1)
                to_annotate_set = loaded_f[start:end] + loaded_f[len(loaded_f) - divide_sets[divide_set_fn][1]:]
                print("{}: {}".format(save_file, len(to_annotate_set)))
                json.dump(to_annotate_set, open(save_file, "w"), indent=2)
    return save_dirs


import glob
# webqsp_save_files, graphqs_save_files = annotate_examples()
for fn in [
    "predicted_bounds_train_webqsp.json",
    "predicted_bounds_dev_webqsp.json",
    "predicted_bounds_test_webqsp.json",
    "predicted_bounds_train_graphqs.json",
    "predicted_bounds_test_graphqs.json",
]:
    print(filter_examples(os.path.join(PARENT_DIR, fn), THRESHOLD))
divide_datasets()