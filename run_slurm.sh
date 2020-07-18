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
# bash run_slurm.sh wiki_all_ents all_avg true 128 false false large 2
# bash run_slurm.sh wiki_all_ents all_avg true 128 false false base 10
data=$1  # wiki_all_ents/webqsp_all_ents/zeshel/pretrain/zero_shot (ledell's model)
mention_agg_type=$2  # all_avg/fl_avg/fl_linear/fl_mlp/none
joint_mention_detection=$3  # "true"/false
context_length=$4  # 16/128
load_saved_cand_encs=$5  # "true"/false
adversarial=$6
model_size=$7
latest_epoch=$8

save_dir="models/entity_encodings/${data}_${mention_agg_type}_${joint_mention_detection}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${output_path_model_size}_${mention_scoring_method}"
# save_dir="models/entity_encodings/${data}_${mention_agg_type}_biencoder_${joint_mention_detection}_${context_length}_${latest_epoch}"

for i in {0..5}
do
    start=$(( i * 10 ))00000
    end=$(( i * 10 + 10 ))00000
    if [ ! -f "${save_dir}/${start}_${end}.t7" ]
    then
        echo ${data} ${mention_agg_type} ${start}
        sbatch examples/train_biencoder.sh \
            ${data} ${mention_agg_type} predict 512 \
            ${joint_mention_detection} ${context_length} \
            ${load_saved_cand_encs} ${adversarial} \
            ${model_size} qa_linear ${start} ${end} ${latest_epoch}
    fi
done
