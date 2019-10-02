# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from multiprocessing.pool import ThreadPool
from candidate_generators import (
    Simple_Candidate_Generator,
    Pregenerated_Candidates_Data_Fetcher,
)

import multiprocessing
import utils
import time
import argparse
import pickle
import os
from evaluator import Evaluator
from tqdm import tqdm

import pysolr
from tqdm import tqdm


def run_thread(arguments):
    mentions = arguments["data"]
    candidate_generator = arguments["candidate_generator"]
    args = arguments["args"]
    if args.keep_pregenerated_candidates:
        data_fetcher = arguments["pregenereted_cands_data_fetcher"]

    if arguments["id"] == 0:
        print("Query args: ", candidate_generator.query_arguments)
        print_query_flag = True
        for mention in tqdm(mentions):
            mention["generated_candidates"] = candidate_generator.get_candidates(
                mention, print_query_flag=print_query_flag
            )
            print_query_flag = False

            if args.keep_pregenerated_candidates:
                wikidata_ids = mention["candidates_wikidata_ids"]
                mention["candidates_data"] = data_fetcher.get_candidates_data(
                    wikidata_ids
                )
    else:
        for mention in mentions:
            mention["generated_candidates"] = candidate_generator.get_candidates(
                mention
            )
            if args.keep_pregenerated_candidates:
                wikidata_ids = mention["candidates_wikidata_ids"]
                mention["candidates_data"] = data_fetcher.get_candidates_data(
                    wikidata_ids
                )

    return arguments["id"], mentions


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def main(args):
    wall_start = time.time()
    parameters = get_parameters(args)

    print("Candidate generator parameters:", parameters)

    datasets = utils.get_datasets(
        args.include_aida_train, args.keep_pregenerated_candidates
    )

    if args.single_dataset:
        datasets = [datasets[0]]

    mentions = utils.get_list_of_mentions(datasets)

    # NUM_TREADS = multiprocessing.cpu_count()
    NUM_THREADS = args.num_threads
    pool = ThreadPool(NUM_THREADS)

    # Split the data into approximately equal parts and give one block to each thread
    data_per_thread = split(mentions, NUM_THREADS)

    if args.keep_pregenerated_candidates:
        arguments = [
            {
                "id": idx,
                "data": data_bloc,
                "args": args,
                "candidate_generator": Simple_Candidate_Generator(parameters),
                "pregenereted_cands_data_fetcher": Pregenerated_Candidates_Data_Fetcher(
                    parameters
                ),
            }
            for idx, data_bloc in enumerate(data_per_thread)
        ]
    else:
        arguments = [
            {
                "id": idx,
                "data": data_bloc,
                "args": args,
                "candidate_generator": Simple_Candidate_Generator(parameters),
            }
            for idx, data_bloc in enumerate(data_per_thread)
        ]

    results = pool.map(run_thread, arguments)

    # Merge the results
    processed_mentions = []
    for _id, mentions in results:
        processed_mentions = processed_mentions + mentions

    has_gold = 0

    pool.terminate()
    pool.join()
    execution_time = (time.time() - wall_start) / 60
    print("The execution took:", execution_time, " minutes")

    # Evaluate the generation
    evaluator = Evaluator(processed_mentions)
    evaluator.candidate_generation(
        save_gold_pos=True, save_pregenerated_gold_pos=args.keep_pregenerated_candidates
    )

    # Dump the data if the dump_mentions flag was set
    if args.dump_mentions:
        print("Dumping processed mentions")
        # Create the directory for the mention dumps if it does not exist
        dump_folder = args.dump_mentions_folder
        os.makedirs(dump_folder, exist_ok=True)

        dump_object = {}
        dump_object["mentions"] = processed_mentions
        dump_object["total_per_dataset"] = evaluator.total_per_dataset
        dump_object["has_gold_per_dataset"] = evaluator.has_gold_per_dataset
        dump_object["parameters"] = parameters
        dump_object["args"] = args
        dump_object["execution_time"] = execution_time

        pickle.dump(
            dump_object,
            open(os.path.join(dump_folder, args.dump_file_id), "wb"),
            protocol=4,
        )

    # evaluator.candidate_generation(max_rank=100)
    return evaluator.recall


def get_parameters(args):
    parameters = {
        "collection_name": args.collection_name,
        "rows": args.rows,
        "solr_address": args.solr_address,
    }

    parameters["query_data"] = {}
    parameters["query_data"]["string"] = args.query
    parameters["query_data"]["keys"] = [k.strip() for k in args.keys.split(",")]
    parameters["boosting"] = args.boosting

    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Debugging setting
    parser.add_argument("--single_dataset", dest="single_dataset", action="store_true")
    parser.set_defaults(single_dataset=False)

    # Query parameters
    parser.add_argument(
        "--query",
        type=str,
        default='title:( {} ) OR aliases:" {} " OR sent_desc_1:( {} )^0.5',
        help="The query following the argument template of q.format",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="mention,mention,sent_context_curr",
        help="The comma separated list of keys to be feeded to str.format with the query as the formating string. Example fields `mention`, `query_context`, `query_truncated_10_context` or `query_truncated_25_context`",
    )

    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--collection_name", type=str, default="wikipedia")
    parser.add_argument("--solr_address", type=str, default="http://localhost:8983")

    parser.add_argument(
        "--boosting", type=str, default="log(sum(num_incoming_links,1))"
    )

    # Multithreading
    parser.add_argument("--num_threads", type=int, required=True)

    # Candidates dumping
    parser.add_argument("--dump_mentions", dest="dump_mentions", action="store_true")
    parser.set_defaults(dump_mentions=False)
    parser.add_argument(
        "--dump_mentions_folder", type=str, default="data/mention_dumps"
    )
    parser.add_argument("--dump_file_id", type=str)

    # Include training dataset
    parser.add_argument(
        "--include_aida_train", dest="include_aida_train", action="store_true"
    )
    parser.set_defaults(include_aida_train=False)

    # Keep pregenerated candidates
    parser.add_argument(
        "--keep_pregenerated_candidates",
        action="store_true",
        help="Whether to keep the candidates given with the dataset.",
    )

    args = parser.parse_args()

    print(args)
    main(args)
