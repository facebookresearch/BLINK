# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class Evaluator:
    def __init__(self, data):
        self.data = data

    def candidate_generation(
        self, max_rank=None, save_gold_pos=False, save_pregenerated_gold_pos=False
    ):
        has_gold_per_dataset = {}
        total_per_dataset = {}
        recall = {}
        processed_mentions = self.data

        if max_rank is None:
            print("Max rank: None")
        else:
            print("Max rank", max_rank)

        for mention in processed_mentions:
            dataset_name = mention["dataset_name"]
            gold_wikidata_id = mention["gold_wikidata_id"]
            gold_pos = -1

            for idx, cand in enumerate(mention["generated_candidates"]):
                cand_wikidata_id = cand["wikidata_id"]
                if gold_wikidata_id == cand_wikidata_id:
                    gold_pos = idx + 1  # Because idx starts at 0
                    break

            if save_gold_pos:
                mention["gold_pos"] = gold_pos

            if gold_pos > 0 and ((max_rank is None) or gold_pos <= max_rank):
                has_gold = has_gold_per_dataset.get(dataset_name, 0) + 1
                has_gold_per_dataset[dataset_name] = has_gold

            if save_pregenerated_gold_pos:
                pre_gen_gold_pos = -1

                for idx, cand in enumerate(mention["candidates_data"]):
                    cand_wikidata_id = cand["wikidata_id"]
                    if gold_wikidata_id == cand_wikidata_id:
                        pre_gen_gold_pos = idx + 1  # Because idx starts at 0
                        break

                mention["pre_gen_candidates_gold_pos"] = pre_gen_gold_pos

            total = total_per_dataset.get(dataset_name, 0) + 1
            total_per_dataset[dataset_name] = total

        total = 0
        has_gold = 0

        for dataset_name in total_per_dataset:
            has_gold_ds = has_gold_per_dataset.get(dataset_name, 0)
            total_ds = total_per_dataset[dataset_name]

            has_gold += has_gold_ds
            total += total_ds

            recall[dataset_name] = has_gold_ds / total_ds
            print("Dataset:", dataset_name)
            print(
                "Recall (w.r.t candidate generation): {:.3f}".format(
                    recall[dataset_name]
                )
            )

        recall["overall"] = has_gold / total
        print(
            "Overal recall (w.r.t candidate generation): {:.3f}".format(
                recall["overall"]
            )
        )

        self.has_gold_per_dataset = has_gold_per_dataset
        self.total_per_dataset = total_per_dataset
        self.total = total
        self.has_gold = has_gold
        self.recall = recall

    def candidate_generation_recall_at(self, ax=None, max_rank=None):
        processed_mentions = self.data
        total_num_of_docs = len(processed_mentions)

        gold_positions = np.array(
            [
                mention["gold_pos"]
                for mention in processed_mentions
                if mention["gold_pos"] >= 0
            ]
        )
        if ax == None:
            fig = plt.figure(figsize=(7, 7))
            ax = plt.subplot(111)
            ax.set_ylabel(str("Recall"))
            ax.set_xlabel(str("True entity rank"))

        rank_count_pairs = sorted(Counter(gold_positions).items(), key=lambda x: x[0])

        # rank_count_pairs = rank_count_pairs[:k]

        counts = [i[1] for i in rank_count_pairs]
        recall = np.cumsum(counts) / total_num_of_docs * 100
        rankings = [i[0] for i in rank_count_pairs]

        if max_rank is not None:
            for idx, rank in enumerate(rankings):
                if rank > max_rank:
                    rankings = rankings[:idx]
                    recall = recall[:idx]
                    break

        ax.plot(rankings, recall)

