from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder, to_bert_input
import blink.candidate_ranking.utils as utils
from blink.main_dense import _get_test_samples, _run_biencoder, _load_candidates, get_mention_bound_candidates, _process_biencoder_dataloader

import json
import logging
from tqdm import tqdm

logger = utils.get_logger(None)
logger.setLevel(10)

entity_catalogue = "models/entity.jsonl"
entity_encoding = "/private/home/belindali/BLINK/models/all_entities_large.t7"
entity_token_ids = "models/entity_token_ids_128.t7"

biencoder_config = "/private/home/belindali/BLINK/models/biencoder_wiki_large.json"
biencoder_model = "/private/home/belindali/BLINK/models/biencoder_wiki_large.bin"
biencoder_wiki, biencoder_params = load_model(biencoder_config, biencoder_model, logger)

# biencoder_config = "/private/home/belindali/pretrain/BLINK-mentions/experiments/webqsp/biencoder_none_false_128_true/training_params.txt"
biencoder_config = "/private/home/belindali/BLINK/models/biencoder_wiki_large.json"
biencoder_model = "/private/home/belindali/pretrain/BLINK-mentions/experiments/webqsp/biencoder_none_false_128_true/pytorch_model.bin"
biencoder_web, biencoder_params = load_model(biencoder_config, biencoder_model, logger)

test_mentions = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.dev.entities.all_pos.filtered_on_all.json"
test_entities = "models/entity.jsonl"

(
    candidate_encoding,
    candidate_token_ids,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    kb2id,
    id2kb,
) = _load_candidates(
    entity_catalogue, entity_encoding, entity_token_ids,
    biencoder_wiki, 128, True,
    logger=logger
)

samples, num_unk, sample_to_all_context_inputs = __load_test(
    test_mentions, #test_entities,
    kb2id=kb2id, logger=logger, qa_classifier_threshold=0,
    qa_data=True, id2kb=id2kb, title2id=title2id,
    do_ner="qa_classifier", debug=False,
    main_entity_only=False,# do_map_test_entities=(len(kb2id) == 0),
    biencoder=biencoder_wiki,
)

dataloader = _process_biencoder_dataloader(
    samples, biencoder_wiki.tokenizer, biencoder_params
)

import torch
import time
for step, batch in enumerate(tqdm(dataloader)):
    context_input, _, label_ids, mention_idxs = batch
    with torch.no_grad():
        import pdb
        pdb.set_trace()
        t1 = time.time()
        scores, start_logits, end_logits = biencoder_wiki.score_candidate(context_input, None, cand_encs=candidate_encoding, gold_mention_idxs=mention_idxs,)
        t2 = time.time()
        scores, start_logits, end_logits = biencoder_web.score_candidate(context_input, None, cand_encs=candidate_encoding, gold_mention_idxs=mention_idxs,)
        t3 = time.time()
        print(t2 - t1)
        print(t3 - t2)


import math
def __load_test(
    test_filename, kb2id, logger, qa_classifier_threshold,
    qa_data=False, id2kb=None, title2id=None,
    do_ner="none", use_ngram_extractor=False, max_mention_len=4,
    debug=False, main_entity_only=False, biencoder=None, saved_ngrams=None,
):
    test_samples = []
    sample_to_all_context_inputs = []  # if multiple mentions found for an example, will have multiple inputs
                                        # maps indices of examples to list of all indices in `samples`
                                        # i.e. [[0], [1], [2, 3], ...]
    unknown_entity_samples = []
    num_unknown_entity_samples = 0
    num_no_gold_entity = 0
    ner_errors = 0
    ner_model = None
    qa_classifier_saved = {}
    if do_ner == "qa_classifier":
        assert qa_classifier_threshold is not None
        if qa_classifier_threshold == "top1":
            do_top_1 = True
        else:
            do_top_1 = False
            qa_classifier_threshold = float(qa_classifier_threshold)
        if "webqsp.test" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/webqsp/test_predictions.json"
        elif "webqsp.dev" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/webqsp/dev_predictions.json"
        elif "graph.test" in test_filename:
            test_predictions_json = "/private/home/sviyer/datasets/graphquestions/test_predictions.json"
        with open(test_predictions_json) as f:
            for line in f:
                line_json = json.loads(line)
                all_ex_preds = []
                for i, pred in enumerate(line_json['all_predictions']):
                    if "graph.test" in test_filename:
                        try:
                            pred['logit'][1] = math.log(pred['logit'][1])
                        except:
                            import pdb
                            pdb.set_trace()
                    if (
                        (do_top_1 and i == 0) or 
                        (not do_top_1 and pred['logit'][1] > qa_classifier_threshold)
                        # or i == 0  # have at least 1 candidate
                    ):
                        all_ex_preds.append(pred)
                assert '1' in line_json['predictions']
                qa_classifier_saved[line_json['id']] = all_ex_preds
    with open(test_filename, "r") as fin:
        if qa_data:
            lines = json.load(fin)
            sample_idx = 0
            do_setup_samples = True
            for i, record in enumerate(tqdm(lines)):
                new_record = {}
                new_record["q_id"] = record["question_id"]
                if main_entity_only:
                    if "main_entity" not in record or record["main_entity"] is None:
                        num_no_gold_entity += 1
                        new_record["label"] = None
                        new_record["label_id"] = -1
                        new_record["all_gold_entities"] = []
                        new_record["all_gold_entities_ids"] = []
                    elif record['main_entity'] in kb2id:
                        new_record["label"] = record["main_entity"]
                        new_record["label_id"] = kb2id[record['main_entity']]
                        new_record["all_gold_entities"] = [record["main_entity"]]
                        new_record["all_gold_entities_ids"] = [kb2id[record['main_entity']]]
                    else:
                        num_unknown_entity_samples += 1
                        unknown_entity_samples.append(record)
                        # sample_to_all_context_inputs.append([])
                        # TODO DELETE?
                        continue
                else:
                    new_record["label"] = None
                    new_record["label_id"] = -1
                    if "entities" not in record or record["entities"] is None or len(record["entities"]) == 0:
                        if "main_entity" not in record or record["main_entity"] is None:
                            num_no_gold_entity += 1
                            new_record["all_gold_entities"] = []
                            new_record["all_gold_entities_ids"] = []
                        else:
                            new_record["all_gold_entities"] = [record["main_entity"]]
                            new_record["all_gold_entities_ids"] = [kb2id[record['main_entity']]]
                    else:
                        new_record["all_gold_entities"] = record['entities']
                        new_record["all_gold_entities_ids"] = []
                        for ent_id in new_record["all_gold_entities"]:
                            if ent_id in kb2id:
                                new_record["all_gold_entities_ids"].append(kb2id[ent_id])
                            else:
                                num_unknown_entity_samples += 1
                                unknown_entity_samples.append(record)
                (new_record_list, sample_idx_list,
                saved_ngrams, ner_errors, sample_idx) = get_mention_bound_candidates(
                    do_ner, record, new_record,
                    saved_ngrams=saved_ngrams, ner_model=ner_model,
                    max_mention_len=max_mention_len, ner_errors=ner_errors,
                    sample_idx=sample_idx, qa_classifier_saved=qa_classifier_saved,
                    biencoder=biencoder,
                )
                if sample_idx_list is not None:
                    sample_to_all_context_inputs.append(sample_idx_list)
                if new_record_list is not None:
                    test_samples += new_record_list
        else:
            lines = fin.readlines()
            for i, line in enumerate(tqdm(lines)):
                record = json.loads(line)
                record["label"] = record["label_id"]
                record["q_id"] = record["query_id"]
                if record["label"] in kb2id:
                    sample_to_all_context_inputs.append([len(test_samples)])
                    record["label_id"] = kb2id[record["label"]]
                    # LOWERCASE EVERYTHING !
                    record["context_left"] = record["context_left"].lower()
                    record["context_right"] = record["context_right"].lower()
                    record["mention"] = record["mention"].lower()
                    record["gold_context_left"] = record["context_left"].lower()
                    record["gold_context_right"] = record["context_right"].lower()
                    record["gold_mention"] = record["mention"].lower()
                    test_samples.append(record)
    # save info and log
    with open("saved_preds/unknown.json", "w") as f:
        json.dump(unknown_entity_samples, f)
    if do_ner == "ngram":
        save_file = "{}_saved_ngrams_new_rules_{}.json".format(test_filename, max_mention_len)
        with open(save_file, "w") as f:
            json.dump(saved_ngrams, f)
        logger.info("Finished saving to {}".format(save_file))
    if logger:
        logger.info("{}/{} samples considered".format(len(sample_to_all_context_inputs), len(lines)))
        logger.info("{} samples generated".format(len(test_samples)))
        logger.info("{} samples with unknown entities considered".format(num_unknown_entity_samples))
        logger.info("{} samples with no gold entities considered".format(num_no_gold_entity))
        logger.info("ner errors: {}".format(ner_errors))
    return test_samples, num_unknown_entity_samples, sample_to_all_context_inputs


def load_model(biencoder_config, biencoder_model, logger):
    # load biencoder model
    logger.info("loading biencoder model")
    try:
        with open(biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(biencoder_config) as json_file:
            for line in json_file:
                line = line.replace("'", "\"")
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                biencoder_params = json.loads(line)
                break
    biencoder_params["path_to_model"] = biencoder_model
    biencoder_params["eval_batch_size"] = 64
    biencoder_params["no_cuda"] = True
    if biencoder_params["no_cuda"]:
        biencoder_params["data_parallel"] = False
    biencoder_params["load_cand_enc_only"] = False
    biencoder = load_biencoder(biencoder_params)
    if type(biencoder.model).__name__ == 'DataParallel':
        biencoder.model = biencoder.model.module
    return biencoder, biencoder_params

