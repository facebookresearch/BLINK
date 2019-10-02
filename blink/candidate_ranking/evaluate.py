# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import utils
import torch
import utils
import argparse
import os

from bert_reranking import BertReranker
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from tqdm import tqdm


def evaluate_model_on_dataset(
    model,
    dataloader,
    dataset_name,
    device,
    logger,
    number_of_samples,
    eval_bm45_acc=False,
    path_to_file_to_write_results=None,
):
    model.eval()

    eval_accuracy = 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids, entity_mask in tqdm(
        dataloader, desc="Evaluating"
    ):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        entity_mask = entity_mask.to(device)

        with torch.no_grad():
            tmp_eval_loss, logits = model(
                input_ids, segment_ids, input_mask, label_ids, entity_mask=entity_mask
            )

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        tmp_eval_accuracy = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    logger.info("\n")

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples

    result = {"normalized_accuracy": normalized_eval_accuracy}
    result["unnormalized_accuracy"] = eval_accuracy / number_of_samples
    result["candidate_generation_recall"] = nb_eval_examples / number_of_samples

    if eval_bm45_acc:
        result["normalized_bm45_recall_@"] = utils.eval_precision_bm45_dataloader(
            dataloader, [1, 5, 10, 20, 40, 60, 80, 100]
        )
        result["unnormalized_bm45_recall_@"] = utils.eval_precision_bm45_dataloader(
            dataloader, [1, 5, 10, 20, 40, 60, 80, 100], number_of_samples
        )

    if path_to_file_to_write_results is None:
        logger.info(
            "***** Eval results - {} ({} / {} samples) *****\n".format(
                dataset_name, nb_eval_examples, number_of_samples
            )
        )
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        with open(path_to_file_to_write_results, "a+") as writer:
            logger.info(
                "***** Eval results - {} ({} / {} samples) *****\n".format(
                    dataset_name, nb_eval_examples, number_of_samples
                )
            )
            writer.write(
                "***** Eval results - {} ({} / {} samples) *****\n".format(
                    dataset_name, nb_eval_examples, number_of_samples
                )
            )
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")

    logger.info("\n")

    return result


def evaluate(parameters, logger=None):
    reranker = utils.get_reranker(parameters)

    if parameters["full_evaluation"]:
        eval_datasets = [
            "aida-A",
            "aida-B",
            "msnbc",
            "aquaint",
            "ace2004",
            "clueweb",
            "wikipedia",
        ]
    else:
        eval_datasets = ["aida-B"]

    candidates_key = (
        "pregenerated_candidates"
        if parameters["evaluate_with_pregenerated_candidates"]
        else "candidates"
    )
    gold_key = (
        "pregenerated_gold_pos"
        if parameters["evaluate_with_pregenerated_candidates"]
        else "gold_pos"
    )

    number_of_samples_per_dataset = {}
    total_time = 0

    for eval_dataset_name in eval_datasets:
        time_start = time.time()
        logger.info("\nEvaluating on the {} dataset".format(eval_dataset_name))
        eval_samples = utils.read_dataset(
            eval_dataset_name, parameters["path_to_preprocessed_json_data"]
        )
        eval_samples_filtered = utils.filter_samples(
            eval_samples, parameters["top_k"], gold_key
        )
        logger.info(
            "Retained {} out of {} samples".format(
                len(eval_samples_filtered), len(eval_samples)
            )
        )
        number_of_samples_per_dataset[eval_dataset_name] = len(eval_samples)

        # if args.num_preprocessing_threads == -1:
        #     eval_data, eval_tensor_data = process_samples_for_model(args.context_key, eval_samples_filtered, tokenizer, args.max_seq_length, logger = logger, top_k = args.top_k, example = False, debug = args.debug, tagged = args.tag_mention, candidates_key = candidates_key, gold_key = gold_key)
        # else:
        #     eval_data, eval_tensor_data = preprocessing_multithreaded(eval_samples_filtered, logger, args, output_dir=True)

        eval_data, eval_tensor_data = reranker._process_mentions_for_model(
            parameters["context_key"],
            eval_samples_filtered,
            reranker.tokenizer,
            parameters["max_seq_length"],
            parameters["top_k"],
            parameters["silent"],
            candidates_key=candidates_key,
            gold_key=gold_key,
            debug=parameters["debug"],
        )

        eval_sampler = SequentialSampler(eval_tensor_data)
        eval_dataloader = DataLoader(
            eval_tensor_data,
            sampler=eval_sampler,
            batch_size=parameters["evaluation_batch_size"],
        )

        if parameters["output_eval_file"] is None:
            output_eval_file = os.path.join(
                parameters["path_to_model"], "eval_results.txt"
            )
        else:
            output_eval_file = parameters["output_eval_file"]

        result = evaluate_model_on_dataset(
            reranker.model,
            eval_dataloader,
            eval_dataset_name,
            eval_bm45_acc=True,
            device=reranker.device,
            logger=logger,
            path_to_file_to_write_results=output_eval_file,
            number_of_samples=number_of_samples_per_dataset[eval_dataset_name],
        )

        execution_time = (time.time() - time_start) / 60
        total_time += execution_time
        if logger != None:
            logger.info(
                "The execution for dataset {} took {} minutes".format(
                    eval_dataset_name, execution_time
                )
            )
        else:
            print(
                "The execution for dataset {} took {} minutes".format(
                    eval_dataset_name, execution_time
                )
            )

    if logger != None:
        logger.info(
            "The execution for dataset {} took {} minutes".format(
                eval_dataset_name, execution_time
            )
        )
    else:
        print("The evaluation took:", total_time, " minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_preprocessed_json_data",
        default="data/train_and_benchmark_processed_json",
        type=str,
        help="The path to the train and benchmarking data.",
    )
    parser.add_argument(
        "--path_to_model",
        default=None,
        type=str,
        required=True,
        help="The full path to the model to be evaluated.",
    )

    parser.add_argument("--top_k", default=80, type=int)
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )

    parser.add_argument(
        "--context_key", default="tagged_query_context_sent_prev_curr_next", type=str
    )
    parser.add_argument(
        "--lowercase_flag",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--full_evaluation",
        action="store_true",
        help="Whether to run the evaluation on all datasets.",
    )
    parser.add_argument(
        "--evaluate_with_pregenerated_candidates",
        action="store_true",
        help="Whether to run in debug mode with only 200 samples.",
    )

    parser.add_argument(
        "--evaluation_batch_size",
        default=8,
        type=int,
        help="Total batch size for evaluation.",
    )
    parser.add_argument(
        "--dataparallel_bert",
        action="store_true",
        help="Whether to distributed the candidate generation process.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode with only 200 samples.",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Whether to print progress bars."
    )

    parser.add_argument(
        "--output_eval_file",
        default=None,
        type=str,
        help="The txt file where the the evaluation results will be written.",
    )

    args = parser.parse_args()
    print(args)

    parameters = args.__dict__
    evaluate(parameters, logger=utils.get_logger())

