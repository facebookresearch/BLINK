# process finetuning data into correct format

# dict_keys(['category', 'text', 'tokenized_text_ids', 'tokenized_mention_idxs', 'mentions', 'text_chunks', 'chunk_idx_to_mention_idx_map', 'label_id', 'wikidata_id', 'entity', 'label'])
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

all_wiki_ents = open("/private/home/ledell/data/wiki_ent2/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_titles_to_ents = {line['title']: i for i, line in enumerate(all_wiki_ents)}
print("Created wikipedia title -> idx in saved entities map")

all_examples = {}
for fn in glob.glob("/checkpoint/fabiopetroni/KILT/backup_datasets/entity_linking/list_entities_formulation/AIDA-YAGO2-*-kilt.jsonl"):
    print(fn)
    store_name = fn[len("/checkpoint/fabiopetroni/KILT/backup_datasets/entity_linking/list_entities_formulation/AIDA-YAGO2-"):-len("-kilt.jsonl")]
    if store_name not in all_examples:
        all_examples[store_name] = []
    with open(fn) as f:
        all_examples[store_name] += f.readlines()

print(len(all_examples))

all_desc_examples = {}
for example_split in all_examples:
    all_desc_examples[example_split] = []
    for example in tqdm(all_examples[example_split]):
        example = json.loads(example)
        all_have_desc = True
        for ent in example["output"]:
            ent = ent['provenance']
            assert len(ent) == 1
            if ent[0]['title'] not in all_wiki_titles_to_ents:
                all_have_desc = False
                break
        if all_have_desc:
            all_desc_examples[example_split].append(example)

print(len(all_desc_examples))

examples_new = {}
for split in all_desc_examples:
    examples_new[split] = {}
    # merge same documents
    for j, example in enumerate(tqdm(all_desc_examples[split])):
        ex_id = example["id"].split(":")[0]
        if ex_id not in examples_new[split]:
            examples_new[split][ex_id] = {
                'id': ex_id,
                'text': '',
                'mentions': [],
                'label_id': [],  #-- ID in all_wiki_titles_to_ents
                'wikidata_id': [],
                'entity': [],  #-- wiki title corresponding to label_id
                'label': [],  #-- wiki descriptions corresponding to label_id
            }
        chunk_id = int(example["id"].split(":")[1])
        all_ents = example["output"]
        label_ids = [all_wiki_titles_to_ents[ent["provenance"][0]["title"]] for ent in all_ents]
        if len(examples_new[split][ex_id]['text']) > 0 and examples_new[split][ex_id]['text'][-1] != ' ':
            examples_new[split][ex_id]['text'] += ' '
        example_ranges = [[
            ent["provenance"][0]["meta"]["input_start_character"] + len(examples_new[split][ex_id]['text']),
            ent["provenance"][0]["meta"]["input_end_character"] + len(examples_new[split][ex_id]['text']),
        ] for ent in all_ents]
        examples_new[split][ex_id]['text'] += example["input"]
        raw_mention_text = [ent["provenance"][0]["meta"]["mention"] for ent in all_ents]
        wikipedia_titles = [all_wiki_ents[_id]['title'] for _id in label_ids]  #-- wiki title corresponding to label_id
        wikipedia_desc = [all_wiki_ents[_id]['text'] for _id in label_ids]  #-- wiki descriptions corresponding to label_id
        wikipedia_wiki_ids = [all_wiki_ents[_id]['kb_idx'] if 'kb_idx' in all_wiki_ents[_id] else None for _id in label_ids]  # wikidata IDs
        # wikipedia_cats = [all_wiki_ents[_id]['catetories'] for _id in label_ids]  #-- wiki categories corresponding to label_id
        # check alignments
        for m in range(len(raw_mention_text)):
            assert examples_new[split][ex_id]['text'][example_ranges[m][0]:example_ranges[m][1]] == raw_mention_text[m]
        examples_new[split][ex_id]['mentions'] += example_ranges
        examples_new[split][ex_id]['label_id'] += label_ids  #-- ID in all_wiki_titles_to_ents
        examples_new[split][ex_id]['wikidata_id'] += wikipedia_wiki_ids
        examples_new[split][ex_id]['entity'] += wikipedia_titles  #-- wiki title corresponding to label_id
        examples_new[split][ex_id]['label'] += wikipedia_desc  #-- wiki descriptions corresponding to label_id


for split in examples_new:
    with open("/checkpoint/belindali/entity_link/data/AIDA-YAGO2/{}.jsonl".format(split), "w") as wf:
        for example in tqdm(examples_new[split]):
            b=wf.write(json.dumps(examples_new[split][example]) + "\n")


# create chunked data, as well as mappings from each chunk to all of its corresponding mentions
examples_new_2 = {}
for split in examples_new:
    examples_new_2[split] = []
    for ex_id in tqdm(examples_new[split]):
        example = examples_new[split][ex_id]
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
                assert chunk == full_example[example_ranges[mention_idx][0]:example_ranges[mention_idx][1]]
        new_ex = {
            'id': example['id'],
            'text': full_example,
            'mentions': example_ranges,
            'text_chunks': example_chunks,
            'chunk_idx_to_mention_idx_map': chunk_idx_to_mention_idx_map,
            'label_id': example['label_id'],  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': example['wikidata_id'],
            'entity': example['entity'],  #-- wiki title corresponding to label_id
            'label': example['label'],
        }
        examples_new_2[split].append(new_ex)


for split in examples_new_2:
    with open("/checkpoint/belindali/entity_link/data/AIDA-YAGO2/{}_chunked.jsonl".format(split), "w") as wf:
        for example in tqdm(examples_new_2[split]):
            b=wf.write(json.dumps(example) + "\n")

examples_new = examples_new_2

# restore from checkpoint
for split in ["train", "test"]:
    file_lines = open("/checkpoint/belindali/entity_link/data/AIDA-YAGO2/{}_chunked.jsonl".format(split)).readlines()
    examples_new[split] = {}
    for line in file_lines:
        line_json = json.loads(line)
        examples_new[split][line_json['id']] = line_json

examples_new_2 = {}
for split in examples_new:
    # for i, split_examples in enumerate(examples_new):
    examples_new_2[split] = []
    for ex_id in tqdm(examples_new[split]):
        example = examples_new[split][ex_id]
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
            'id': example['id'],
            'text': example['text'],
            'mentions': example['mentions'],
            'tokenized_text_ids': all_token_ids,
            'tokenized_mention_idxs': mention_idxs,
            'label_id': example['label_id'],  #-- ID in all_wiki_titles_to_ents
            'wikidata_id': example['wikidata_id'],
            'entity': example['entity'],  #-- wiki title corresponding to label_id
            'label': example['label'],
        }
        examples_new_2[split].append(new_ex)


num_long = []
for split in examples_new_2:
    with open("/checkpoint/belindali/entity_link/data/AIDA-YAGO2/tokenized/{}.jsonl".format(split), "w") as wf:
        for i, example in tqdm(enumerate(examples_new_2[split])):
            if len(example['tokenized_text_ids']) > 512:
                num_long.append(i)
            b=wf.write(json.dumps(example) + "\n")
