# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using ELQ.
import argparse
import importlib
import os
import sys
import datetime


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


class ElqParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_elq_args:
        (default True) initializes the default arguments for ELQ package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_elq_args=True, add_model_args=False, 
        description='ELQ parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_elq_args,
        )
        self.elq_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['ELQ_HOME'] = self.elq_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_elq_args:
            self.add_elq_args()
        if add_model_args:
            self.add_model_args()

    def add_elq_args(self, args=None):
        """
        Add common ELQ args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            default=True,
            type=bool,
            help="Whether the dataset is from zeroshot.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_false",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument("--context_key", default="context", type=str)
        parser.add_argument("--title_key", default="entity", type=str)
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )
        parser.add_argument(
            "--mention_aggregation_type",
            default=None,
            type=str,
            help="Type of mention aggregation (None to just use [CLS] token, "
            "'all_avg' to average across tokens in mention, 'fl_avg' to average across first/last tokens in mention, "
            "'{all/fl}_linear' for linear layer over mention, '{all/fl}_mlp' to MLP over mention)",
        )
        parser.add_argument(
            "--no_mention_bounds",
            dest="no_mention_bounds",
            action="store_true",
            default=False,
            help="Don't add tokens around target mention. MUST BE FALSE IF 'mention_aggregation_type' is NONE",
        )
        parser.add_argument(
            "--mention_scoring_method",
            dest="mention_scoring_method",
            default="qa_linear",
            type=str,
            help="Method for generating/scoring mentions boundaries (options: 'qa_mlp', 'qa_linear', 'BIO')",
        )
        parser.add_argument(
            "--max_mention_length",
            dest="max_mention_length",
            default=10,
            type=int,
            help="Maximum length of span to consider as candidate mention",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="Total batch size for training."
        )
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=5, 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=40,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, 
            help="Whether to shuffle train data",
        )
        # TODO DELETE LATER!!!
        parser.add_argument(
            "--start_idx",
            default=None,
            type=int,
        )
        parser.add_argument(
            "--end_idx",
            default=None,
            type=int,
        )
        parser.add_argument(
            "--last_epoch",
            default=0,
            type=int,
            help="Epoch to restore from when pretraining",
        )
        parser.add_argument(
            "--path_to_trainer_state",
            default=None,
            type=str,
            required=False,
            help="The full path to the last checkpoint's training state to load.",
        )
        parser.add_argument(
            '--dont_distribute_train_samples',
            default=False,
            action="store_true",
            help="Don't distribute all training samples across the epochs (go through all samples every epoch)",
        )
        parser.add_argument(
            "--freeze_cand_enc",
            default=False,
            action="store_true",
            help="Freeze the candidate encoder",
        )
        parser.add_argument(
            "--load_cand_enc_only",
            default=False,
            action="store_true",
            help="Only load the candidate encoder from saved model path",
        )
        parser.add_argument(
            "--cand_enc_path",
            default="models/all_entities_large.t7",
            type=str,
            required=False,
            help="Filepath to the saved entity encodings.",
        )
        parser.add_argument(
            "--cand_token_ids_path",
            default="models/entity_token_ids_128.t7",
            type=str,
            required=False,
            help="Filepath to the saved tokenized entity descriptions.",
        )
        parser.add_argument(
            "--index_path",
            default="models/faiss_hnsw_index.pkl",
            type=str,
            required=False,
            help="Filepath to the HNSW index for adversarial training.",
        )
        parser.add_argument(
            "--adversarial_training",
            default=False,
            action="store_true",
            help="Do adversarial training (only takes effect if `freeze_cand_enc` is set)",
        )
        parser.add_argument(
            "--get_losses",
            default=False,
            action="store_true",
            help="Get losses during evaluation",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for candidate pool",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for candidate encoding",
        )
