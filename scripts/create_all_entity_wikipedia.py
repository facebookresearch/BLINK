import json
import glob
from time import time
import random
from tqdm import tqdm
import string
import numpy as np

from pytorch_transformers.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True,
)

'''
from kilt.knowledge_source import KnowledgeSource
import json
from tqdm import tqdm
# get the knowledge souce
ks = KnowledgeSource()

# all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
all_wiki_ents = open("/private/home/ledell/data/wiki_ent2/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_titles_to_ents = {line['title']: i for i, line in enumerate(all_wiki_ents)}
print("Created wikipedia title -> idx in saved entities map")

dev_examples=open("/checkpoint/belindali/NQ/nq-dev-multi-new-new.json").readlines()
dev_examples = [json.loads(line) for line in dev_examples]

split=0
num_answers = 0
saved_wiki_examples = open("/checkpoint/belindali/NQ/out_wiki_{}.txt".format(split), "w")
in_wiki_answers = []
no_wiki_answers = []
for i, dev_ex in enumerate(tqdm(dev_examples[int(split * 1000): int((split + 1) * 1000)])):
    # if i < len(in_wiki_answers) + len(no_wiki_answers):
    #     continue
    for answer in dev_ex['answers']:
        num_answers += 1
        if answer in all_wiki_titles_to_ents:
            in_wiki_answers.append((i, dev_ex['question'], answer))
            continue
        page = ks.get_page_by_title(answer)
        if page is None:
            no_wiki_answers.append((i, dev_ex['question'], answer))
            b = saved_wiki_examples.write(str(i) + "\n")
        else:
            in_wiki_answers.append((i, dev_ex['question'], answer))

print(num_answers)
json.dump(no_wiki_answers, open("/checkpoint/belindali/NQ/out_wiki_{}.json".format(split), "w"))

all_splits = []
last_num = 0
for split in [0, 2, 3, 4]:
    print("/checkpoint/belindali/NQ/out_wiki_{}.json".format(split))
    saved_wiki_examples = json.load(open("/checkpoint/belindali/NQ/out_wiki_{}.json".format(split)))
    for ex in saved_wiki_examples:
        ex_id = int(split * 1000) + ex[0]
        if ex_id >= last_num:
            last_num = ex_id
            all_splits.append([ex_id, dev_examples[ex_id]['question'], ex[-1]])
        else:
            print(ex_id)

json.dump(all_splits, open("/checkpoint/belindali/NQ/first_4_splits_examples.json", "w"))

# with open("/checkpoint/belindali/NQ/first_4_splits_examples.tsv", "w") as wf:
#     for ex in all_splits:
#         b=wf.write("\t".join([str(e) for e in ex]) + "\n")


import string
import re
four_digits = re.compile('\d\d\d\d')

def check_date(cand):
    cand_lower = cand.lower()
    for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]:
        if month in cand_lower:
            return True
    if "century" in cand_lower:
        return True
    if four_digits.search(cand_lower) is not None:
        return True
    if "BC" in cand:
        return True
    return False

def check_any(cand, query):
    # if any char in query in cand, return true
    for c in query:
        if c in cand:
            return True
    return False

dates = []
numbers = []
no_uppercase = []
other = []
for ex in all_splits:
    answer = ex[-1]
    if check_date(answer):
        dates.append(ex)
    elif check_any(answer, string.digits):
        numbers.append(ex)
    elif not check_any(answer, string.ascii_uppercase):
        no_uppercase.append(ex)
    else:
        other.append(ex)

# [578, 168, 246, 850] -- 2158
# 14.45%, 4.2%, 6.15%,
all_lists = [dates, numbers, no_uppercase, other]
save_files = ["/checkpoint/belindali/NQ/first_4_splits_dates.tsv", "/checkpoint/belindali/NQ/first_4_splits_nums.tsv", "/checkpoint/belindali/NQ/first_4_splits_lowers.tsv", "/checkpoint/belindali/NQ/first_4_splits_other.tsv"]
for i, sf in enumerate(save_files):
    with open(sf, "w") as wf:
        for ex in all_lists[i]:
            b = wf.write("\t".join([str(e) for e in ex]) + "\n")

# '''


NUM_TRAIN = 9000000
NUM_DEV = 10000
NUM_TEST = 10000

# all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
all_wiki_ents = open("/private/home/ledell/data/wiki_ent2/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_titles_to_ents = {line['title']: i for i, line in enumerate(all_wiki_ents)}
print("Created wikipedia title -> idx in saved entities map")

# [u'category', u'context_left', u'entity', u'mention', u'label_id', u'context_right',
# u'label': wiki article
# ]
# [u'wikidata_id', u'title', u'context_left', u'label', u'mention', u'label_id', u'context_right', u'entity', u'question_id']

all_examples = []
for fn in glob.glob("/private/home/ledell/data/wikipedia_raw/processed/*.json"):
    print(fn)
    with open(fn) as f:
        all_examples += f.readlines()

print(len(all_examples))

all_desc_examples = []
for example in tqdm(all_examples):
    example = json.loads(example)
    all_have_desc = True
    for ent in example["ent"]:
        if ent[0] not in all_wiki_titles_to_ents:
            all_have_desc = False
            break
    if all_have_desc:
        valid_examples.append(example)

print(len(all_desc_examples))

json.dump(all_desc_examples, open("/checkpoint/belindali/entity_link/data/all_samples_with_desc.json", "w"))
# with open("/checkpoint/belindali/entity_link/data/all_samples_with_desc.jsonl", "w") as wf:
#     for example in tqdm(all_desc_examples):
#         b=wf.write(json.dumps(all_desc_examples) + "\n")

chosen_examples = random.sample(all_desc_examples, NUM_TRAIN + NUM_DEV + NUM_TEST)

# train_examples = []
# valid_examples = []
# test_examples = []
train_examples = open("/checkpoint/belindali/entity_link/data/train.jsonl").readlines()
train_examples = [json.loads(example) for example in tqdm(train_examples)]
valid_examples = open("/checkpoint/belindali/entity_link/data/valid.jsonl").readlines()
test_examples = open("/checkpoint/belindali/entity_link/data/test.jsonl").readlines()
valid_examples = [json.loads(example) for example in tqdm(valid_examples)]
test_examples = [json.loads(example) for example in tqdm(test_examples)]

examples_new = [train_examples, valid_examples, test_examples]
for i, split_examples in enumerate([train_examples, valid_examples, test_examples]):
    for j, example in enumerate(tqdm(split_examples)):
        all_ents = example["ent"]
        label_ids = [all_wiki_titles_to_ents[ent[0]] for ent in all_ents]
        all_spans = [[ent[1], ent[2]] for ent in all_ents]
        wikipedia_titles = [all_wiki_ents[_id]['title'] for _id in label_ids]  #-- wiki title corresponding to label_id
        wikipedia_desc = [all_wiki_ents[_id]['text'] for _id in label_ids]  #-- wiki descriptions corresponding to label_id
        wikipedia_wiki_ids = [all_wiki_ents[_id]['kb_idx'] if 'kb_idx' in all_wiki_ents[_id] else None for _id in label_ids]  # wikidata IDs
        wikipedia_cats = [all_wiki_ents[_id]['catetories'] for _id in label_ids]  #-- wiki categories corresponding to label_id
        new_ex = {
            'context_left': [example['text'][:span[0]] for span in all_spans],
            'mention': [example['text'][span[0]:span[1]+1] for span in all_spans],
            'context_right': [example['text'][span[1]+1:] for span in all_spans],
        }
        # shift to align with words (whitespace)
        for m in range(len(new_ex['mention'])):
            while len(new_ex['context_right'][m]) > 0 and new_ex['context_right'][m][0] in in_mention_letters:
                assert new_ex['context_right'][m][0] not in ignore
                new_ex['mention'][m] += new_ex['context_right'][m][0]
                new_ex['context_right'][m] = new_ex['context_right'][m][1:]
            while len(new_ex['context_left'][m]) > 0 and new_ex['context_left'][m][-1] in in_mention_letters:
                assert new_ex['context_left'][m][-1] not in ignore
                new_ex['mention'][m] = new_ex['context_left'][m][-1:] + new_ex['mention'][m]
                new_ex['context_left'][m] = new_ex['context_left'][m][:-1]
        new_ex = {
            'category': wikipedia_cats,
            'context_left': new_ex['context_left'],
            'mention': new_ex['mention'],
            'context_right': new_ex['context_right'],
            'label_id': label_ids,  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': wikipedia_wiki_ids,
            'entity': wikipedia_titles,  #-- wiki title corresponding to label_id
            'label': wikipedia_desc,
        }
        examples_new[i].append(new_ex)
    # if i < NUM_TRAIN:
    #     train_examples.append(new_ex)
    # elif i < NUM_TRAIN + NUM_DEV:
    #     valid_examples.append(new_ex)
    # else:
    #     test_examples.append(new_ex)

'''
# in_mention_letters = string.ascii_letters + string.digits
# ignore = string.punctuation + string.whitespace
# special_chars = {}
import copy
examples_new_2 = [[], [], []]
for i, split_examples in enumerate(examples_new):
    for j, example in enumerate(tqdm(split_examples)):
        full_example = example['context_left'][0] + example['mention'][0] + example['context_right'][0]
        example_ranges = []
        for m in range(len(example['mention'])):
            # whitespace around each example...
            mention_range = [len(example['context_left'][m]), len(example['context_left'][m]) + len(example['mention'][m])]
            example_ranges.append(mention_range)
            assert full_example[mention_range[0]:mention_range[1]] == example['mention'][m], "not = mention"
        orig_example_ranges = copy.deepcopy(example_ranges)
        num_spaces_added_first = 0
        num_spaces_added = 0
        for m, mention_range in enumerate(example_ranges):
            if m > 0 and example_ranges[m][0] < orig_example_ranges[m-1][1] and num_spaces_added > 0:
                assert example_ranges[m][0] >= orig_example_ranges[m-1][0]
                example_ranges[m][0] += num_spaces_added_first
                example_ranges[m][1] += num_spaces_added_first
            else:
                example_ranges[m][0] += num_spaces_added
                example_ranges[m][1] += num_spaces_added
            new_full_example = full_example[:example_ranges[m][0]].rstrip() + " " + full_example[example_ranges[m][0]:].lstrip()
            if len(new_full_example) != len(full_example):
                assert len(new_full_example) - len(full_example) < 2, "difference is {}".format(len(new_full_example) - len(full_example))
                assert len(new_full_example) - len(full_example) > -2, "difference is {}".format(len(new_full_example) - len(full_example))
                num_spaces_added += (len(new_full_example) - len(full_example))
                num_spaces_added_first = num_spaces_added
                example_ranges[m][0] += (len(new_full_example) - len(full_example))
                example_ranges[m][1] += (len(new_full_example) - len(full_example))
            full_example = new_full_example
            new_full_example = full_example[:example_ranges[m][1]].rstrip() + " " + full_example[example_ranges[m][1]:].lstrip()
            if len(new_full_example) != len(full_example):
                assert len(new_full_example) - len(full_example) < 2, "difference is {}".format(len(new_full_example) - len(full_example))
                assert len(new_full_example) - len(full_example) > -2, "difference is {}".format(len(new_full_example) - len(full_example))
                num_spaces_added += (len(new_full_example) - len(full_example))
            full_example = new_full_example
            if full_example[example_ranges[m][0]] == " ":
                example_ranges[m][0] += 1
            if full_example[example_ranges[m][1] - 1] == " ":
                example_ranges[m][1] -= 1
            if full_example[example_ranges[m][0]:example_ranges[m][1]] != example['mention'][m].strip(string.punctuation + string.whitespace) and full_example[example_ranges[m][0]:example_ranges[m][1]] != example['mention'][m]:
                import pdb
                pdb.set_trace()
        new_ex = {
            'category': example['category'],
            'context_left': example['context_left'],
            'mention': example['mention'],
            'context_right': example['context_right'],
            'label_id': example['label_id'],  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': example['wikidata_id'],
            'entity': example['entity'],  #-- wiki title corresponding to label_id
            'label': example['label'],
        }
        examples_new_2[i].append(new_ex)
# '''

# create chunked data, as well as mappings from each chunk to all of its corresponding mentions
examples_new_2 = [[], [], []]
for i, split_examples in enumerate(examples_new):
    for j, example in enumerate(tqdm(split_examples)):
        full_example = example['text']
        example_ranges = example['mentions']
        # flatten and sort boundaries
        char_in_mention_idx_map = [[] for _ in range(len(full_example))]
        all_mention_bounds = []
        for m, ment in enumerate(example_ranges):
            for c in range(ment[0], ment[1]):
                char_in_mention_idx_map[c].append(m)
            all_mention_bounds.append(ment[0])
            all_mention_bounds.append(ment[1])
        all_mention_bounds = [0] + all_mention_bounds + [len(full_example)]
        all_mention_bounds = list(set(all_mention_bounds))
        all_mention_bounds.sort()
        # create chunks
        example_chunks = [full_example[all_mention_bounds[b]:(all_mention_bounds[b+1])] for b in range(len(all_mention_bounds) - 1)]
        # chunk_idx : [list of mentions the chunk is part of]
        chunk_idx_to_mention_idx_map = []
        bound_idx = 0
        for c, chunk in enumerate(example_chunks):
            assert bound_idx == all_mention_bounds[c]
            try:
                chunk_idx_to_mention_idx_map.append(char_in_mention_idx_map[all_mention_bounds[c]])
            except:
                print("error checkpoint")
                import pdb
                pdb.set_trace()
            bound_idx += len(chunk)
        # TODO check chunks
        for chunk_idx, mention_idx_list in enumerate(chunk_idx_to_mention_idx_map):
            chunk = example_chunks[chunk_idx]
            for mention_idx in mention_idx_list:
                assert chunk in full_example[example_ranges[mention_idx][0]:example_ranges[mention_idx][1]]
        new_ex = {
            'category': example['category'],
            'text': full_example,
            'mentions': example_ranges,
            'text_chunks': example_chunks,
            'chunk_idx_to_mention_idx_map': chunk_idx_to_mention_idx_map,
            'label_id': example['label_id'],  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': example['wikidata_id'],
            'entity': example['entity'],  #-- wiki title corresponding to label_id
            'label': example['label'],
        }
        examples_new_2[i].append(new_ex)

examples_new = examples_new_2

# chunked data -> mention boundary data for tokenized version
split_idx = 5
splits = [0, 1500000, 3000000, 4500000, 6000000, 7500000, 9000001]
train_examples = train_examples[splits[split_idx]:splits[split_idx+1]]
train_examples = [json.loads(example) for example in tqdm(train_examples)]
split_examples = train_examples
examples_new_2 = []
with open("/checkpoint/belindali/entity_link/data/tokenized/train_{}_{}.jsonl".format(splits[split_idx], splits[split_idx+1]), "w") as wf:
    # for i, split_examples in enumerate(examples_new):
    for j, example in enumerate(tqdm(split_examples)):
        example_chunks = example['text_chunks']
        chunk_idx_to_mention_idx_map = example['chunk_idx_to_mention_idx_map']
        mention_idx_to_chunk_idx_map = {}
        chunk_idx_to_tokenized_bounds = {}
        mention_idxs = []
        all_token_ids = []
        # tokenize chunks and set any mention boundaries
        cum_len = 0
        for c, chunk in enumerate(example_chunks):
            chunk_tokens = tokenizer.encode(chunk)
            all_token_ids += chunk_tokens
            chunk_bounds = [cum_len, cum_len+len(chunk_tokens)]
            for m in chunk_idx_to_mention_idx_map[c]:
                if m not in mention_idx_to_chunk_idx_map:
                    mention_idx_to_chunk_idx_map[m] = chunk_bounds
                else:
                    existing_chunk_bounds = mention_idx_to_chunk_idx_map[m]
                    mention_idx_to_chunk_idx_map[m] = [
                        min(existing_chunk_bounds[0], chunk_bounds[0]),
                        max(existing_chunk_bounds[1], chunk_bounds[1]),
                    ]
            cum_len += len(chunk_tokens)
        # convert to list
        for mention_idx in range(len(mention_idx_to_chunk_idx_map)):
            assert mention_idx in mention_idx_to_chunk_idx_map
            mention_tokenized_bound = mention_idx_to_chunk_idx_map[mention_idx]
            mention_idxs.append(mention_tokenized_bound)
        for m in range(len(mention_idxs)):
            mention_bounds = example['mentions'][m]
            mention_tok_bounds = mention_idxs[m]
            tokenized_mention = tokenizer.decode(all_token_ids[
                mention_tok_bounds[0]:mention_tok_bounds[1]
            ])
            target_mention = example['text'][mention_bounds[0]:mention_bounds[1]].lower()
            try:
                assert tokenized_mention == target_mention
            except:
                # only keep letters and whitespace
                only_letter_tokenized_mention = ""
                only_letter_target_mention = ""
                for char in tokenized_mention:
                    if char in string.ascii_letters:
                        only_letter_tokenized_mention += char
                for char in target_mention:
                    if char in string.ascii_letters:
                        only_letter_target_mention += char
        new_ex = {
            'category': example['category'],
            'tokenized_text_ids': all_token_ids,
            'tokenized_mention_idxs': mention_idxs,
            'label_id': example['label_id'],  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': example['wikidata_id'],
            'entity': example['entity'],  #-- wiki title corresponding to label_id
            'label': example['label'],
        }
        examples_new_2.append(new_ex)
        b=wf.write(json.dumps(new_ex) + "\n")

# for example in examples_new_2:
#     if len(example['tokenized_text_ids']) > 512:
#         tokenizer.decode(example['tokenized_text_ids'])
#         for mention in example['tokenized_mention_idxs']:
#             tokenizer.decode(example['tokenized_text_ids'][mention[0]:mention[1]])
#             import pdb
#             pdb.set_trace()

# save train.jsonl/valid.jsonl/test.jsonl
with open("/checkpoint/belindali/entity_link/data/tokenized/train_{}_{}.jsonl".format(start,end), "w") as wf:
    for example in tqdm(examples_new_2):
        b=wf.write(json.dumps(example) + "\n")

with open("/checkpoint/belindali/entity_link/data/tokenized/valid.jsonl", "w") as wf:
    for example in tqdm(examples_new_2[1]):
        b=wf.write(json.dumps(example) + "\n")

with open("/checkpoint/belindali/entity_link/data/tokenized/test.jsonl", "w") as wf:
    for example in tqdm(examples_new_2[2]):
        b=wf.write(json.dumps(example) + "\n")

all_train_examples = []
for s in range(len(splits) - 1):
    train_examples = open("/checkpoint/belindali/entity_link/data/tokenized/train_{}_{}.jsonl".format(splits[s],splits[s+1])).readlines()
    train_examples = [json.loads(example) for example in tqdm(train_examples)]
    all_train_examples += train_examples

with open("/checkpoint/belindali/entity_link/data/tokenized/train.jsonl", "w") as wf:
    for example in tqdm(all_train_examples):
        b=wf.write(json.dumps(example) + "\n")

# Random sampling
# loaded_json = {
#     "category": 
# }

# train_wiki = open("/private/home/ledell/data/wiki_ent2/train.jsonl").readlines()
# dev_wiki = open("/private/home/ledell/data/wiki_ent2/valid.jsonl").readlines()
# test_wiki = open("/private/home/ledell/data/wiki_ent2/test.jsonl").readlines()

# print("read lines")

# start = time()
# train_wiki = [json.loads(line) for line in train_wiki]
# dev_wiki = [json.loads(line) for line in dev_wiki]
# test_wiki = [json.loads(line) for line in test_wiki]
# end = time()

# print("loaded jsons (took {0}s)".format(end - start))
# [u'title', u'context_left', u'label', u'mention', u'label_id', u'context_right', u'entity']

# for wiki in [train_wiki, dev_wiki, test_wiki]:
#     for entry in wiki:
#         ks.get_page_by_id(entry["label_id"])
#         ks.get_page_by_id(entry["label_id"])
#         page = ks.get_page_by_title(entry["entity"])