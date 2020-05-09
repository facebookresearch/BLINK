import torch
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.eval_biencoder import load_or_generate_candidate_pool, encode_candidate
import blink.candidate_ranking.utils as utils
import json
import sys
import os

import argparse


# TO replicate Ledell's encodings...
'''
python scripts/generate_candidates.py --path_to_model_config /private/home/belindali/BLINK/models/biencoder_wiki_large.json \
    --path_to_model /private/home/belindali/BLINK/models/biencoder_wiki_large.bin \
    --entity_dict_path /private/home/belindali/BLINK/models/entity.jsonl \
    --compare_saved_embeds /private/home/belindali/BLINK/models/all_entities_large.t7 \
    --saved_cand_ids /private/home/belindali/BLINK/models/entity_token_ids_128.t7 --test
'''

parser = argparse.ArgumentParser()
# /private/home/belindali/temp/BLINK-Internal/models/biencoder_wiki_large.json
# /private/home/ledell/BLINK-Internal/models/biencoder_wiki.json
parser.add_argument('--path_to_model_config', type=str, required=True, help='filepath to saved model config')
# /private/home/belindali/temp/BLINK-Internal/models/biencoder_wiki_large.bin
# /checkpoint/ledell/20191126/train_hard_negatives_wiki_top10/learningrate=5e-05/model/epoch_27_0/pytorch_model.bin
parser.add_argument('--path_to_model', type=str, required=True, help='filepath to saved model')
# /private/home/belindali/temp/BLINK-Internal/models/tac_entity.jsonl
# /private/home/belindali/temp/BLINK-Internal/models/entity.jsonl
# /private/home/ledell/BLINK-Internal/models/toy.jsonl
parser.add_argument('--entity_dict_path', type=str, required=True, help='filepath to entities to encode (.jsonl file)')
# /private/home/belindali/temp/BLINK-Internal/models/entity_token_ids.t7
parser.add_argument('--saved_cand_ids', type=str, help='filepath to entities pre-parsed into IDs')
# /checkpoint/belindali/entity_link_models/saved/candidate_encodes/webqsp_data_bienc_finetune/my_train_try{n}
parser.add_argument('--encoding_save_file_dir', type=str, help='directory of file to save generated encodings', default=None)
parser.add_argument('--test', action='store_true', default=False, help='whether to just test encoding subsample of entities')

# /private/home/belindali/temp/BLINK-Internal/models/tac_candidate_encode_large.t7
# /private/home/belindali/temp/BLINK-Internal/models/all_entities_large.t7
# /private/home/ledell/BLINK-Internal/models/tac_candidate_encode.t7
parser.add_argument('--compare_saved_embeds', type=str, help='compare against these saved embeddings')

parser.add_argument('--batch_size', type=int, default=512, help='batch size for encoding candidate vectors (default 512)')

parser.add_argument('--chunk_start', type=int, default=0, help='example idx to start encoding at (for parallelizing encoding process)')
parser.add_argument('--chunk_end', type=int, default=-1, help='example idx to stop encoding at (for parallelizing encoding process)')


args = parser.parse_args()

try:
    with open(args.path_to_model_config) as json_file:
        biencoder_params = json.load(json_file)
except json.decoder.JSONDecodeError:
    with open(args.path_to_model_config) as json_file:
        for line in json_file:
            line = line.replace("'", "\"")
            line = line.replace("True", "true")
            line = line.replace("False", "false")
            line = line.replace("None", "null")
            biencoder_params = json.loads(line)
            break
# model to use
#'path_to_model': '/private/home/belindali/BLINK-Internal/data/experiments/biencoder/trained_on_webqsp/pytorch_model.bin',   # WEBQSP TRAINED MODEL
#'path_to_model': '/private/home/belindali/BLINK-Internal/models/biencoder_wiki.bin',  # ZESHEL MODEL COPIED FROM REPO
#'/private/home/belindali/BLINK-Internal/data/experiments/biencoder/pytorch_model.bin',   #  ZESHEL BIENCODER MODEL I TRAINED
biencoder_params["path_to_model"] = args.path_to_model
# entities to use
#'/private/home/belindali/BLINK-Internal/models/entity.jsonl',  # TACKBP DATA 
#'entity_dict_path': '/private/home/belindali/BLINK-Internal/models/wdt_entities.jsonl',  # WEBQSP DATA
biencoder_params["entity_dict_path"] = args.entity_dict_path
biencoder_params["degug"] = False
biencoder_params["data_parallel"] = True
biencoder_params["no_cuda"] = False
biencoder_params["max_context_length"] = 32
biencoder_params["encode_batch_size"] = args.batch_size

# biencoder_params = {'data_path': '', 'bert_model': 'bert-base-uncased', 'model_output_path': None, 'context_key': 'context', 'lowercase': True, 'top_k': 10, 'max_seq_length': 128, 'evaluate': False, 'evaluate_with_pregenerated_candidates': False, 'output_eval_file': None, 'debug': False, 'silent': False, 'train_batch_size': 8, 'eval_batch_size': 128, 'data_parallel': True, 'max_grad_norm': 1.0, 'learning_rate': 3e-05, 'num_train_epochs': 1, 'print_interval': 5, 'eval_interval': 40, 'save_interval': 1, 'warmup_proportion': 0.1, 'no_cuda': False, 'seed': 52313, 'gradient_accumulation_steps': 1, 'out_dim': 100, 'pull_from_layer': -1, 'type_optimization': 'all_encoder_layers', 'add_linear': False, 'shuffle': False, 'encode_batch_size': 1024, 'max_context_length': 128, 'is_zeshel': False, 'degug': False, 'max_cand_length': 128,
#     'path_to_model': args.path_to_model,
#     'entity_dict_path': args.entity_dict_path,
# }
saved_cand_ids = getattr(args, 'saved_cand_ids', None)
#"/private/home/belindali/BLINK-Internal/models/entity_ids_{}.t7".format(
#        biencoder_params["max_cand_length"])  # TACKBP DATA
#saved_cand_ids = "/private/home/belindali/BLINK-Internal/models/wdt_entity_ids_{}.t7".format(
#        biencoder_params["max_cand_length"])  # WEBQSP DATA
encoding_save_file_dir = args.encoding_save_file_dir
if encoding_save_file_dir is not None and not os.path.exists(encoding_save_file_dir):
    os.makedirs(encoding_save_file_dir, exist_ok=True)
#encoding_save_file_pref = "/private/home/belindali/BLINK-Internal/models/entities_seqlen_{}_my_train.t7".format(
#        biencoder_params["max_cand_length"])
#encoding_save_file_pref = "/private/home/belindali/BLINK-Internal/models/entities_seqlen_{}_my_train.t7".format(
#        biencoder_params["max_cand_length"])

logger = utils.get_logger(biencoder_params.get("model_output_path", None))
biencoder = load_biencoder(biencoder_params)
baseline_candidate_encoding = None
if getattr(args, 'compare_saved_embeds', None) is not None:
    baseline_candidate_encoding = torch.load(getattr(args, 'compare_saved_embeds'))
# if args.test:
#     assert biencoder_params['entity_dict_path'][len(biencoder_params['entity_dict_path'])-6:] == '.jsonl'
#     new_entity_dict_path = biencoder_params['entity_dict_path'][:len(biencoder_params['entity_dict_path'])-6]
#     new_entity_dict_path = '{}_sample.jsonl'.format(new_entity_dict_path)
#     with open(biencoder_params['entity_dict_path']) as f:
#         with open(new_entity_dict_path, 'w') as wf:
#             for i, line in enumerate(f):
#                 wf.write(line)
#                 if i > 10:
#                     break
#     biencoder_params['entity_dict_path'] = new_entity_dict_path

candidate_pool = load_or_generate_candidate_pool(
    biencoder.tokenizer,
    biencoder_params,
    logger,
    getattr(args, 'saved_cand_ids', None),
)
if args.test:
    candidate_pool = candidate_pool[:10]

# encode in chunks to parallelize
save_file = None
if getattr(args, 'encoding_save_file_dir', None) is not None:
    save_file = os.path.join(
        args.encoding_save_file_dir,
        "{}_{}.t7".format(args.chunk_start, args.chunk_end),
    )
print("Saving in: {}".format(save_file))
# if os.path.exists(save_file):
#     print("Save file already exists!: {}".format(save_file))
#     sys.exit()

if save_file is not None:
    f = open(save_file, "w").close()  # mark as existing

candidate_encoding = encode_candidate(
    biencoder,
    candidate_pool[args.chunk_start:args.chunk_end],
    biencoder_params["encode_batch_size"],
    biencoder_params["silent"],
    logger,
)

if save_file is not None:
    torch.save(candidate_encoding, save_file)

print(candidate_encoding[0,:10])
if baseline_candidate_encoding is not None:
    print(baseline_candidate_encoding[0,:10])

