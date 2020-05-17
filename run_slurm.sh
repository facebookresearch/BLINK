#!/bin/sh
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=USR1
#SBATCH --mem=200000
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time 3000
#SBATCH --constraint=volta32gb

# bash run_slurm.sh pretrain all_avg true 128 60
data=$1  # webqsp/zeshel/pretrain/zero_shot (ledell's model)
mention_agg_type=$2  # all_avg/fl_avg/fl_linear/fl_mlp/none
joint_mention_detection=$3  # "true"/false
context_length=$4  # 16/128
latest_epoch=$5

# for i in {0..11}
# do
#     start=$(( i * 5 ))00000
#     end=$(( i * 5 + 5 ))00000
#     if [ ! -f "models/entity_encodings/${data}_${mention_agg_type}_biencoder_10/${start}_${end}.t7" ]
#     then
#         echo ${data} ${mention_agg_type} ${start}
#         sbatch examples/train_biencoder.sh ${data} ${mention_agg_type} predict 512 ${start} ${end}
#     fi
# done


for i in {0..3}
do
    start=$(( i * 15 ))00000
    end=$(( i * 15 + 15 ))00000
    if [ ! -f "models/entity_encodings/${data}_${mention_agg_type}_biencoder_${joint_mention_detection}_${context_length}_${latest_epoch}/${start}_${end}.t7" ]
    then
        echo ${data} ${mention_agg_type} ${start}
        sbatch examples/train_biencoder.sh ${data} ${mention_agg_type} predict 512 ${joint_mention_detection} ${context_length} ${start} ${end} ${latest_epoch}
    fi
done
