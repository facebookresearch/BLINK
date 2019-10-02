# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

import blink.utils as utils
import blink.ner as NER
import blink.candidate_generation as CG
import blink.candidate_data_fetcher as CDF
import blink.reranker as R

import argparse
import shutil


def main(parameters):
    print("Parameters:", parameters)
    # Read data
    sentences = utils.read_sentences_from_file(
        parameters["path_to_input_file"],
        one_sentence_per_line=parameters["one_sentence_per_line"],
    )

    # Identify mentions
    ner_model = NER.get_model(parameters)
    ner_output_data = ner_model.predict(sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]

    output_folder_path = parameters["output_folder_path"]

    if (
        (output_folder_path is not None)
        and os.path.exists(output_folder_path)
        and os.listdir(output_folder_path)
    ):
        print(
            "The given output directory ({}) already exists and is not empty.".format(
                output_folder_path
            )
        )
        answer = input("Would you like to empty the existing directory? [Y/N]\n")

        if answer.strip() == "Y":
            print("Deleting {}...".format(output_folder_path))
            shutil.rmtree(output_folder_path)
        else:
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(
                    output_folder_path
                )
            )

    if output_folder_path is not None:
        utils.write_dicts_as_json_per_line(
            sentences, utils.get_sentences_txt_file_path(output_folder_path)
        )
        utils.write_dicts_as_json_per_line(
            mentions, utils.get_mentions_txt_file_path(output_folder_path)
        )

    # Generate candidates and get the data that describes the candidates
    candidate_generator = CG.get_model(parameters)
    candidate_generator.process_mentions_for_candidate_generator(
        sentences=sentences, mentions=mentions
    )

    for mention in mentions:
        mention["candidates"] = candidate_generator.get_candidates(mention)
        if parameters["consider_additional_datafetcher"]:
            data_fetcher = CDF.get_model(parameters)
            for candidate in mention["candidates"]:
                data_fetcher.get_data_for_entity(candidate)

    if output_folder_path is not None:
        utils.write_dicts_as_json_per_line(
            mentions, utils.get_mentions_txt_file_path(output_folder_path)
        )

    # Reranking
    reranking_model = R.get_model(parameters)
    reranking_model.rerank(mentions, sentences)

    if output_folder_path is not None:
        utils.write_dicts_as_json_per_line(
            mentions, utils.get_mentions_txt_file_path(output_folder_path)
        )
        utils.write_end2end_pickle_output(sentences, mentions, output_folder_path)
        utils.present_annotated_sentences(
            sentences,
            mentions,
            utils.get_end2end_pretty_output_file_path(output_folder_path),
        )

    # Showcase results
    utils.present_annotated_sentences(sentences, mentions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument(
        "--path_to_input_file",
        "--i",
        dest="path_to_input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--one_sentence_per_line",
        action="store_true",
        help="Set if the input file has one sentence per line",
    )

    # Candidate generation
    parser.add_argument(
        "--solr_address",
        default="http://localhost:8983/solr/wikipedia",
        type=str,
        help="The address to the solr index.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default='title:( {} ) OR aliases:" {} " OR sent_desc_1:( {} )^0.5',
        help="The query following the argument template of str.format",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="text,text,context",
        help="The comma separated list of keys to be feeded to str.format with the query as the formating string.",
    )
    parser.add_argument(
        "--boosting",
        default="log(sum(num_incoming_links,1))",
        type=str,
        help="The address to the solr index.",
    )
    parser.add_argument(
        "--raw_solr_fields",
        action="store_true",
        help="Whether to escape the special characters in the solr queries.",
    )

    # Candidate desciptions and additional data
    parser.add_argument(
        "--consider_additional_datafetcher",
        action="store_true",
        help="Whether to include some additional data to the candidates using a datafetcher.",
    )
    parser.add_argument(
        "--path_to_candidate_data_dict",
        default="data/KB_data/title2enriched_parsed_obj_plus.p",
        type=str,
        help="The path to the data used by the data fetcher (the default path points to the wikipedia data).",
    )

    # Reranking
    parser.add_argument(
        "--path_to_model",
        "--m",
        dest="path_to_model",
        type=str,
        required=True,
        help="The full path to the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--evaluation_batch_size",
        default=1,
        type=int,
        help="Total batch size for evaluation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=80,
        help="The number of candidates retrieved by the candiadate generator and considered by the reranker",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether to use CUDA when available"
    )
    parser.add_argument(
        "--lowercase_flag",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--context_key",
        default="tagged_context",
        type=str,
        help="The field that contains the mention context.",
    )
    parser.add_argument(
        "--dataparallel_bert",
        action="store_true",
        help="Whether to distributed the candidate generation process.",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Whether to print progress bars."
    )

    # Output
    parser.add_argument(
        "--output_folder_path",
        "--o",
        dest="output_folder_path",
        default=None,
        type=str,
        help="A path to the folder where the mentions and sentences are to be dumped. If it is not given, the results would not be saved.",
    )

    args = parser.parse_args()
    args.rows = args.top_k
    parameters = args.__dict__
    main(parameters)
