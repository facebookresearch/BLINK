import argparse
import json
import logging
import os
import random
import time
import torch
from datetime import timedelta


WORLDS = {
    'american_football',
    'doctor_who',
    'fallout',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
}

domain_set = {}
domain_set['val'] = set(['coronation_street', 'muppets', 'ice_hockey', 'elder_scrolls'])
domain_set['test'] = set(['forgotten_realms', 'lego', 'star_trek', 'yugioh'])
domain_set['train'] = set(['american_football', 'doctor_who', 'fallout', 'final_fantasy', 'military', 'pro_wrestling', 'starwars', 'world_of_warcraft'])

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


log_formatter = LogFormatter()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.handlers = []
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(console_handler)


def load_entity_dict(params):
    entity_dict = {}
    entity_map = {}
    for src in WORLDS:
        fname = os.path.join(params.document_path, src + ".json")
        assert os.path.isfile(fname), "File not found! %s" % fname
        cur_dict = {}
        doc_map = {}
        doc_list = []
        with open(fname, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                doc_id = item["document_id"]
                title = item["title"]
                text = item["text"]
                doc_map[doc_id] = len(doc_list)
                doc_list.append(item)

        logger.info("Load for world %s." % src)
        entity_dict[src] = doc_list
        entity_map[src] = doc_map

    return entity_dict, entity_map


def convert_data(params, entity_dict, entity_map, mode):
    if mode == "valid":
        fname = os.path.join(params.mention_path, "val.json")
    else:
        fname = os.path.join(params.mention_path, mode + ".json")

    fout = open(os.path.join(params.output_path, mode + ".jsonl"), 'wt')
    cnt = 0
    max_tok = 128
    with open(fname, 'rt') as f:
        for line in f:
            cnt += 1
            line = line.rstrip()
            item = json.loads(line)
            mention = item["text"].lower()
            src = item["corpus"]
            label_doc_id = item["label_document_id"]
            orig_doc_id = item["context_document_id"]
            start = item["start_index"]
            end = item["end_index"]

            # add context around the mention as well
            orig_id = entity_map[src][orig_doc_id]
            text = entity_dict[src][orig_id]["text"].lower()
            tokens = text.split(" ")

            assert mention == ' '.join(tokens[start:end + 1]) 
            tokenized_query = mention

            mention_context_left = tokens[max(0, start - max_tok):start]
            mention_context_right = tokens[end + 1:min(len(tokens), end + max_tok + 1)]

            # entity info
            k = entity_map[src][label_doc_id]
            ent_title = entity_dict[src][k]['title']
            ent_text = entity_dict[src][k]["text"]

            example = {}
            example["context_left"] = ' '.join(mention_context_left)
            example['context_right'] = ' '.join(mention_context_right)
            example["mention"] = mention
            example["label"] = ent_text
            example["label_id"] = k
            example['label_title'] = ent_title
            example['world'] = src
            fout.write(json.dumps(example))
            fout.write('\n')

    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot Entity Linking Dataset')
    parser.add_argument(
        '--document_path', 
        default='data/zeshel/documents',
        type=str,
    )
    parser.add_argument(
        '--mention_path', 
        default='data/zeshel/mentions',
        type=str,
    )
    parser.add_argument(
        '--output_path',
        default='data/zeshel/blink_format',
        type=str,
    )
    params = parser.parse_args()
    os.makedirs(params.output_path, exist_ok=True)

    entity_dict, entity_map = load_entity_dict(params)
    convert_data(params, entity_dict, entity_map, 'train')
    convert_data(params, entity_dict, entity_map, 'valid')
    convert_data(params, entity_dict, entity_map, 'test')
