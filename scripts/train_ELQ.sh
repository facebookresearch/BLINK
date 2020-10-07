# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
# user-friendly wrapper around elq_slurm_scripts/train_elq.sh

objective=$1
data=$2  # filepath to tokenized data directory (should have a `train.jsonl`)
max_context_length=$3
train_batch_size=$4
eval_batch_size=$5
epoch=$6  # model checkpoint to pick up from
base_data=$7  # filepath of base pretrained model's data directory
base_epoch=$8  # which epoch of base pretrained model to use

if [ "${epoch}" = "" ] || [ "${epoch}" = "_" ]
then
    epoch=-1
fi

bash elq_slurm_scripts/train_elq.sh $data all_avg $objective $train_batch_size $max_context_length true true large qa_linear 0 -1 $epoch $eval_batch_size $base_data $base_epoch

