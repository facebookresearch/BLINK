# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BEGIN_ENT_TOKEN = "[START_ENT]"
END_ENT_TOKEN = "[END_ENT]"

url2id_cache = {}


def _read_url(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
        soup = BeautifulSoup(html, features="html.parser")
        title = soup.title.string.replace(" - Wikipedia", "").strip()
    return title


def _get_pageid_from_api(title, client=None):
    pageid = None

    title_html = title.strip().replace(" ", "%20")
    url = "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(
        title_html
    )

    try:
        # Package the request, send the request and catch the response: r
        r = requests.get(url)

        # Decode the JSON data into a dictionary: json_data
        json_data = r.json()

        if len(json_data["query"]["pages"]) > 1:
            print("WARNING: more than one result returned from wikipedia api")

        for _, v in json_data["query"]["pages"].items():
            pageid = v["pageid"]
    except:
        pass

    return pageid


def extract_questions(filename):

    # all the datapoints
    global_questions = []

    # left context so far in the document
    left_context = []

    # working datapoints for the document
    document_questions = []

    # is the entity open
    open_entity = False

    # question id in the document
    question_i = 0

    with open(filename) as fin:
        lines = fin.readlines()

        for line in tqdm(lines):

            if "-DOCSTART-" in line:
                # new document is starting

                doc_id = line.split("(")[-1][:-2]

                # END DOCUMENT

                # check end of entity
                if open_entity:
                    document_questions[-1]["input"].append(END_ENT_TOKEN)
                    open_entity = False

                """
                #DEBUG
                for q in document_questions:
                    pp.pprint(q)
                    input("...")
                """

                # add sentence_questions to global_questions
                global_questions.extend(document_questions)

                # reset
                left_context = []
                document_questions = []
                question_i = 0

            else:
                split = line.split("\t")
                token = split[0].strip()

                if len(split) >= 5:
                    B_I = split[1]
                    mention = split[2]
                    #  YAGO2_entity = split[3]
                    Wikipedia_URL = split[4]
                    Wikipedia_ID = split[5]
                    # Freee_base_id = split[6]

                    if B_I == "I":
                        pass

                    elif B_I == "B":

                        title = Wikipedia_URL.split("/")[-1].replace("_", " ")

                        if Wikipedia_ID == "000":

                            if Wikipedia_URL in url2id_cache:
                                pageid = url2id_cache[Wikipedia_URL]
                            else:

                                pageid = _get_pageid_from_api(title)
                                url2id_cache[Wikipedia_URL] = pageid
                            Wikipedia_ID = pageid

                        q = {
                            "id": "{}:{}".format(doc_id, question_i),
                            "input": left_context.copy() + [BEGIN_ENT_TOKEN],
                            "mention": mention,
                            "Wikipedia_title": title,
                            "Wikipedia_URL": Wikipedia_URL,
                            "Wikipedia_ID": Wikipedia_ID,
                            "left_context": left_context.copy(),
                            "right_context": [],
                        }
                        document_questions.append(q)
                        open_entity = True
                        question_i += 1

                    else:
                        print("Invalid B_I {}", format(B_I))
                        sys.exit(-1)

                    # print(token,B_I,mention,Wikipedia_URL,Wikipedia_ID)
                else:
                    if open_entity:
                        document_questions[-1]["input"].append(END_ENT_TOKEN)
                        open_entity = False

                left_context.append(token)
                for q in document_questions:
                    q["input"].append(token)

                for q in document_questions[:-1]:
                    q["right_context"].append(token)

                if len(document_questions) > 0 and not open_entity:
                    document_questions[-1]["right_context"].append(token)

    # FINAL SENTENCE
    if open_entity:
        document_questions[-1]["input"].append(END_ENT_TOKEN)
        open_entity = False

    # add sentence_questions to global_questions
    global_questions.extend(document_questions)

    return global_questions


# store on file
def store_questions(questions, OUT_FILENAME):

    if not os.path.exists(os.path.dirname(OUT_FILENAME)):
        try:
            os.makedirs(os.path.dirname(OUT_FILENAME))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(OUT_FILENAME, "w+") as fout:
        for q in questions:
            json.dump(q, fout)
            fout.write("\n")


def convert_to_BLINK_format(questions):
    data = []
    for q in questions:
        datapoint = {
            "context_left": " ".join(q["left_context"]).strip(),
            "mention": q["mention"],
            "context_right": " ".join(q["right_context"]).strip(),
            "query_id": q["id"],
            "label_id": q["Wikipedia_ID"],
            "Wikipedia_ID": q["Wikipedia_ID"],
            "Wikipedia_URL": q["Wikipedia_URL"],
            "Wikipedia_title": q["Wikipedia_title"],
        }
        data.append(datapoint)
    return data


# AIDA-YAGO2
print("AIDA-YAGO2")
in_aida_filename = (
    "data/train_and_benchmark_data/basic_data/test_datasets/AIDA/AIDA-YAGO2-dataset.tsv"
)
aida_questions = extract_questions(in_aida_filename)

train = []
testa = []
testb = []
for element in aida_questions:
    if "testa" in element["id"]:
        testa.append(element)
    elif "testb" in element["id"]:
        testb.append(element)
    else:
        train.append(element)
print("train: {}".format(len(train)))
print("testa: {}".format(len(testa)))
print("testb: {}".format(len(testb)))

train_blink = convert_to_BLINK_format(train)
testa_blink = convert_to_BLINK_format(testa)
testb_blink = convert_to_BLINK_format(testb)

out_train_aida_filename = "data/BLINK_benchmark/AIDA-YAGO2_train.jsonl"
store_questions(train_blink, out_train_aida_filename)
out_testa_aida_filename = "data/BLINK_benchmark/AIDA-YAGO2_testa.jsonl"
store_questions(testa_blink, out_testa_aida_filename)
out_testb_aida_filename = "data/BLINK_benchmark/AIDA-YAGO2_testb.jsonl"
store_questions(testb_blink, out_testb_aida_filename)


# ACE 2004
print("ACE 2004")
in_ace_filename = "data/train_and_benchmark_data/basic_data/test_datasets/wned-datasets/ace2004/ace2004.conll"
ace_questions = convert_to_BLINK_format(extract_questions(in_ace_filename))
out_ace_filename = "data/BLINK_benchmark/ace2004_questions.jsonl"
store_questions(ace_questions, out_ace_filename)
print(len(ace_questions))


# aquaint
print("aquaint")
in_aquaint_filename = "data/train_and_benchmark_data/basic_data/test_datasets/wned-datasets/aquaint/aquaint.conll"
aquaint_questions = convert_to_BLINK_format(extract_questions(in_aquaint_filename))
out_aquaint_filename = "data/BLINK_benchmark/aquaint_questions.jsonl"
store_questions(aquaint_questions, out_aquaint_filename)
print(len(aquaint_questions))

#  clueweb - WNED-CWEB (CWEB)
print("clueweb - WNED-CWEB (CWEB)")
in_clueweb_filename = "data/train_and_benchmark_data/basic_data/test_datasets/wned-datasets/clueweb/clueweb.conll"
clueweb_questions = convert_to_BLINK_format(extract_questions(in_clueweb_filename))
out_clueweb_filename = "data/BLINK_benchmark/clueweb_questions.jsonl"
store_questions(clueweb_questions, out_clueweb_filename)
print(len(clueweb_questions))


# msnbc
print("msnbc")
in_msnbc_filename = "data/train_and_benchmark_data/basic_data/test_datasets/wned-datasets/msnbc/msnbc.conll"
msnbc_questions = convert_to_BLINK_format(extract_questions(in_msnbc_filename))
out_msnbc_filename = "data/BLINK_benchmark/msnbc_questions.jsonl"
store_questions(msnbc_questions, out_msnbc_filename)
print(len(msnbc_questions))


# wikipedia - WNED-WIKI (WIKI)
print("wikipedia - WNED-WIKI (WIKI)")
in_wnedwiki_filename = "data/train_and_benchmark_data/basic_data/test_datasets/wned-datasets/wikipedia/wikipedia.conll"
wnedwiki_questions = convert_to_BLINK_format(extract_questions(in_wnedwiki_filename))
out_wnedwiki_filename = "data/BLINK_benchmark/wnedwiki_questions.jsonl"
store_questions(wnedwiki_questions, out_wnedwiki_filename)
print(len(wnedwiki_questions))
