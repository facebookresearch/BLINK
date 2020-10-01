# coding: utf-8
# Copyright (C) 2018 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

# Embeddings and vocabulary utility methods

import logging
import re
import os
import codecs

import json
import numpy as np
from pycorenlp import StanfordCoreNLP
from typing import Set, Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"
epsilon = 10e-8

special_tokens = {"&ndash;": "–",
                  "&mdash;": "—",
                  "@card@": "0"
                  }
digits_pattern = re.compile(r"([0-9][0-9.,]*)")

corenlp = StanfordCoreNLP('http://100.97.69.173:9000')
corenlp_properties = {
    'annotators': 'tokenize, pos, ner',
    'outputFormat': 'json'
}
corenlp_caseless = {
    'pos.model': 'edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger',
    'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz,'
}

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)
RESOURCES_FOLDER = "/private/home/belindali/starsem2018-entity-linking/resources/"

split_pattern = re.compile(r"[\s'-:,]")


def load_resource_file_backoff(f):
    def load_method(file_name):
        try:
            return f(file_name)
        except Exception as ex:
            logger.error("No file found. {}".format(ex))
        return None
    return load_method


@load_resource_file_backoff
def load_json_resource(path_to_file):
    with open(path_to_file) as f:
        resource = json.load(f)
    return resource


@load_resource_file_backoff
def load_property_labels(path_to_property_labels):
    """

    :param path_to_property_labels:
    :return:
    >>> load_property_labels("../resources/properties_with_labels.txt")["P106"]
    {'type': 'wikibase-item', 'altlabel': ['employment', 'craft', 'profession', 'job', 'work', 'career'], 'freq': 2290043, 'label': 'occupation'}
    """
    with open(path_to_property_labels, encoding="utf-8") as infile:
        return_map = {}
        for l in infile.readlines():
            if not l.startswith("#"):
                columns = l.split("\t")
                return_map[columns[0].strip()] = {"label": columns[1].strip().lower(),
                                                  "altlabel": list(set(columns[3].strip().lower().split(", "))),
                                                  "type":  columns[4].strip().lower(),
                                                  "freq": int(columns[5].strip().replace(",",""))}
    return return_map


@load_resource_file_backoff
def load_entity_freq_map(path_to_map):
    """
    Load the map of entity frequencies from a file.

    :param path_to_map: location of the map file
    :return: entity map as a dictionary
    >>> load_entity_freq_map("../resources/wikidata_entity_freqs.map")['Q76']
    7070
    """
    return_map = {}
    with open(path_to_map, encoding="utf-8") as f:
        for l in f.readlines():
            k = l.strip().split("\t")
            return_map[k[0]] = int(k[1])
    return return_map


@load_resource_file_backoff
def load_list(path_to_list):
    with open(path_to_list, encoding="utf-8") as f:
        return_list = {l.strip() for l in f.readlines()}
    return return_list


corenlp_pos_tagset = load_list(RESOURCES_FOLDER + "PENN.pos.tagset")
stop_words_en = load_list(RESOURCES_FOLDER + "english.stopwords")


def get_tagged_from_server(input_text, caseless=False) -> Iterable:
    """
    Get pos tagged and ner from the CoreNLP Server. 

    :param input_text: input text as a string
    :return: tokenized text with pos and ne tags
    >>> get_tagged_from_server("Light explodes over Pep Guardiola's head in Bernabeu press room. Will Mourinho stop at nothing?! Heh heh")[0] == \
    {'characterOffsetBegin': 0, 'ner': 'O', 'pos': 'JJ', 'characterOffsetEnd': 5, 'originalText': 'Light', 'lemma': 'light'}
    True
    """
    if len(input_text.strip()) == 0:
        return []
    # input_text = remove_links(input_text)
    input_text = _preprocess_corenlp_input(input_text)
    if caseless:
        input_text = input_text.lower()
    corenlp_output = corenlp.annotate(input_text,
                                      properties={**corenlp_properties, **corenlp_caseless} if caseless else corenlp_properties
                                      ).get("sentences", [])
    tagged = [{k: t[k] for k in {"index", "pos", "ner", "lemma", "characterOffsetBegin", "characterOffsetEnd", "word"}}
              for sent in corenlp_output for t in sent['tokens']]
    return tagged


def _preprocess_corenlp_input(input_text):
    input_text = input_text.replace("/", " / ")
    input_text = input_text.replace("-", " - ")
    input_text = input_text.replace("–", " – ")
    input_text = input_text.replace("_", " _ ")
    return input_text


def lemmatize_tokens(entity_tokens):
    """
    Lemmatize the list of tokens using the Stanford CoreNLP.

    :param entity_tokens:
    :return:
    >>> lemmatize_tokens(['House', 'Of', 'Representatives'])
    ['House', 'Of', 'Representative']
    >>> lemmatize_tokens(['Canadians'])
    ['Canadian']
    >>> lemmatize_tokens(['star', 'wars'])
    ['star', 'war']
    >>> lemmatize_tokens(['Movie', 'does'])
    ['Movie', 'do']
    >>> lemmatize_tokens("who is the member of the house of representatives?".split())
    ['who', 'be', 'the', 'member', 'of', 'the', 'house', 'of', 'representative', '?']
    """
    try:
        lemmas = corenlp.annotate(" ".join([t.lower() for t in entity_tokens]), properties={
            'annotators': 'tokenize, lemma',
            'outputFormat': 'json'
        }).get("sentences", [])[0]['tokens']
    except:
        lemmas = []
    lemmas = [t['lemma'] for t in lemmas]
    lemmas = [l.title() if i < len(entity_tokens) and entity_tokens[i].istitle() else l for i, l in enumerate(lemmas)]
    return lemmas


def load_word_embeddings(path):
    """
    Loads pre-trained embeddings from the specified path.

    @return (embeddings as an numpy array, word to index dictionary)
    """
    word2idx = {}  # Maps a word to the index in the embeddings matrix
    embeddings = []

    with codecs.open(path, 'r', encoding='utf-8') as fIn:
        idx = 1
        for line in fIn:
            split = line.strip().split(' ')
            embeddings.append([float(num) for num in split[1:]])
            word2idx[split[0]] = idx
            idx += 1

    word2idx[all_zeroes] = 0
    embedding_size = len(embeddings[0])
    embeddings = np.asarray([[0.0]*embedding_size] + embeddings, dtype='float32')

    rare_w_ids = list(range(idx-1001, idx-1))
    unknown_emb = np.average(embeddings[rare_w_ids,:], axis=0)
    embeddings = np.append(embeddings, [unknown_emb], axis=0)
    word2idx[unknown_el] = idx
    idx += 1

    logger.debug("Loaded: {}".format(embeddings.shape))

    return embeddings, word2idx


def load_kb_embeddings(path_to_folder):
    """
    Loads pre-trained KB embeddings from the specified path.

    @return (embeddings as an numpy array, relation embeddings, entity2idx, relation2idx)
    """

    entity2idx = {}
    allowed_indices = set()
    with open(path_to_folder + "/entity2id.filtered.txt", 'r') as f:
        f.readline()
        for l in f.readlines():
            k, v, idx = tuple(l.strip().split("\t"))
            entity2idx[k] = int(idx) + 1
            allowed_indices.add(int(v))

    relation2idx = {}
    with open(path_to_folder + "/relation2id.txt", 'r') as f:
        f.readline()
        for l in f.readlines():
            k, v = tuple(l.strip().split("\t"))
            relation2idx[k] = int(v) + 1

    embeddings = []
    relation_embeddings = []
    with open(path_to_folder + "/entity2vec.vec", 'r') as f:
        idx = 0
        for line in f.readlines():
            if idx in allowed_indices:
                split = line.strip().split('\t')
                embeddings.append([float(num) for num in split])
            idx += 1

    entity2idx[all_zeroes] = 0
    embedding_size = len(embeddings[0])
    embeddings = np.asarray([[0.0]*embedding_size] + embeddings, dtype='float32')

    with open(path_to_folder + "/relation2vec.vec", 'r') as f:
        for line in f:
            split = line.strip().split('\t')
            relation_embeddings.append([float(num) for num in split])

    relation2idx[all_zeroes] = 0
    embedding_size = len(relation_embeddings[0])
    relation_embeddings = np.asarray([[0.0]*embedding_size] + relation_embeddings, dtype='float32')

    logger.debug("KB embeddings loaded: {}, {}".format(embeddings.shape, relation_embeddings.shape))

    return embeddings, entity2idx, relation_embeddings, relation2idx


def get_word_idx(word, word2idx):
    """
    Get the word index for the given word. Maps all numbers to 0, lowercases if necessary.

    :param word: the word in question
    :param word2idx: dictionary constructed from an embeddings file
    :return: integer index of the word
    """
    unknown_idx = word2idx[unknown_el]
    word = word.strip()
    if word in word2idx:
        return word2idx[word]
    elif word.lower() in word2idx:
        return word2idx[word.lower()]
    elif word in special_tokens:
        return word2idx.get(special_tokens[word], unknown_idx)
    no_digits = digits_pattern.sub('0', word)
    if no_digits in word2idx:
        return word2idx[no_digits]
    return unknown_idx


def create_elements_index(element_set: Set):
    """
    Create an element to index mapping, that includes a zero and an unknown element.

    :param element_set: set of elements to enumerate
    :return: an index as a dictionary
    >>> create_elements_index({"a", "b", "c", all_zeroes})["_UNKNOWN"] 
    4
    """
    element_set = element_set - {all_zeroes, unknown_el}
    element_set = sorted(list(element_set))
    el2idx = {c: i for i, c in enumerate(element_set, 1)}
    el2idx[all_zeroes] = 0
    el2idx[unknown_el] = len(el2idx)
    return el2idx


def map_pos(pos):
    if pos.endswith("S") or pos.endswith("R"):
        return pos[:-1]
    return pos


def lev_distance(s1, s2, costs=(1, 1, 2)):
    """
    Levinstein distance with adjustable costs

    :param s1: first string
    :param s2: second string
    :param costs: a tuple of costs: (remove, add, substitute)
    :return: a distance as an integer number
    >>> lev_distance("Obama", "Barack Obama", costs=(1,0,1))
    0
    >>> lev_distance("Obama", "Barack Obama", costs=(0,2,1))
    14
    >>> lev_distance("Obama II", "Barack Obama", costs=(1,0,1))
    3
    >>> lev_distance("Chile", "Tacna", costs=(2,1,2))
    10
    >>> lev_distance("Chile", "Chilito", costs=(2,1,2))
    4
    """

    len1 = len(s1)
    len2 = len(s2)
    a_cost, b_cost, c_cost = costs
    lev = [[0] * (len2+1) for _ in range(len1+1)]
    if a_cost > 0:
        for i, v in enumerate(list(range(0, len1*a_cost+1, a_cost))):
            lev[i][0] = v
    if b_cost > 0:
        lev[0] = list(range(0, len2*b_cost+1, b_cost))
    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            c1 = s1[i]
            c2 = s2[j]
            a = lev[i][j+1] + a_cost  # skip character in s1 -> remove
            b = lev[i+1][j] + b_cost  # skip character in s2 -> add
            c = lev[i][j] + (c_cost if c1 != c2 else 0)  # substitute
            lev[i+1][j+1] = a if a < b and a < c else (b if b < c else c)
    return lev[-1][-1]


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
