#!/bin/sh
#SBATCH --output=stdout/%j.out
#SBATCH --error=stderr/%j.err
#SBATCH --partition=priority
#SBATCH --comment=emnlpdeadline06/01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=USR1
#SBATCH --mem=400000
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time 3000
#SBATCH --constraint=volta32gb

# for mention_agg_type in all_avg fl_avg fl_linear fl_mlp none
# do
#   for data in webqsp zeshel
#   do
#     echo ${data} ${mention_agg_type}
#     sbatch examples/train_biencoder.sh ${data} ${mention_agg_type} both
#   done
# done

# for mention_agg_type in all_avg fl_avg
# do
#   for data in webqsp zeshel
#   do
#     for i in {0..5}
#     do
#       echo ${data} ${mention_agg_type} ${i}000000
#       sbatch examples/train_biencoder.sh ${data} ${mention_agg_type} predict 512 ${i}000000 $(( i + 1 ))000000
#     done
#   done
# done


# for i in {1..5}
# do
#   echo ${data} ${mention_agg_type} ${i}000000
#   sbatch examples/train_biencoder.sh ${data} ${mention_agg_type} predict 512 ${i}000000 $(( i + 1 ))000000
# done
# python scripts/merge_candidates.py \
# --path_to_saved_chunks /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder

# sbatch examples/train_biencoder.sh webqsp none both 128
# sbatch examples/train_biencoder.sh webqsp none predict 512
data=$1  # webqsp/zeshel/pretrain_wiki
mention_agg_type=$2  # all_avg/fl_avg/fl_linear/fl_mlp/none
objective=$3  # train/predict/both (default)
batch_size=$4  # 64 (for training large model)
chunk_start=$5
chunk_end=$6

export PYTHONPATH=.

# Example to run bi-encoder on zero-shot entity linking data
# Remove --debug flag to run on full dataset
# Set --data_parallel to run it on multiple GPUs
# Increase num_train_epochs to get better models (i.e. 5)

if [ "${data}" = "webqsp" ]
then
  data_path="/private/home/belindali/starsem2018-entity-linking/data/WebQSP"
elif [ "${data}" = "zeshel" ]
then
  data_path="/private/home/ledell/zeshel/data/biencoder/"
fi

if [ "${mention_agg_type}" != "none" ]
then
  all_mention_args="--no_mention_bounds \
  --mention_aggregation_type ${mention_agg_type}"
else
  all_mention_args=""
fi

if [ "${batch_size}" = "" ]
then
  batch_size="256"
fi

if [ "${objective}" = "" ]
then
  objective="both"
fi

if [ "${objective}" = "both" ] || [ "${objective}" = "train" ]
then
  echo "Running ${mention_agg_type} biencoder training on ${data} dataset."
  if [ "${data}" = "pretrain_wiki" ]
  then
    python blink/biencoder/train_biencoder.py \
      --output_path data/experiments/pretrain/biencoder_${mention_agg_type} \
      --data_path /private/home/ledell/data/wiki_ent2 \
      --num_train_epochs 100 \
      --learning_rate 0.00001 \
      --train_batch_size ${batch_size} \
      --eval_batch_size ${batch_size} \
      --bert_model bert-large-uncased \
      --data_parallel ${all_mention_args}
      #--debug \
      # --start_idx ${chunk_start} --end_idx ${chunk_end}   # TODO DELETE THIS LATER!!!!!
  else
    python blink/biencoder/train_biencoder.py \
      --output_path data/experiments/${data}/biencoder_${mention_agg_type} \
      --path_to_model /private/home/ledell/BLINK-Internal/models/biencoder_wiki_large.bin \
      --data_path ${data_path} \
      --num_train_epochs 5 \
      --learning_rate 0.00001 \
      --max_context_length 256 \
      --max_cand_length 256 \
      --train_batch_size ${batch_size} \
      --eval_batch_size ${batch_size} \
      --bert_model bert-large-uncased \
      --data_parallel ${all_mention_args}
  fi
fi

# echo "Running ${mention_agg_type} biencoder full evaluation on ${data} dataset."
# python blink/biencoder/eval_biencoder.py \
#   --path_to_model data/experiments/${data}/biencoder_${mention_agg_type}/pytorch_model.bin \
#   --data_path ${data_path} \
#   --output_path data/experiments/nn_preds \
#   --encode_batch_size ${batch_size} \
#   --bert_model bert-large-uncased

if [ "${objective}" = "both" ] || [ "${objective}" = "predict" ]
then
  if [ "${chunk_start}" = "" ]
  then
    chunk_start="0"
  fi
  if [ "${chunk_end}" = "" ]
  then
    chunk_end="1000000"
  fi

  if [ ! -f "/private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/training_params.txt" ]
  then
    echo "copying training params to /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/training_params.txt"
    cp data/experiments/${data}/biencoder_${mention_agg_type}/training_params.txt /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/training_params.txt
  fi

  if [ ! -f "/private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/pytorch_model.bin" ]
  then
    echo "copying saved model bin to /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/pytorch_model.bin"
    cp data/experiments/${data}/biencoder_${mention_agg_type}/pytorch_model.bin /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder/pytorch_model.bin
  fi

  echo "Getting ${mention_agg_type}_${data} biencoder candidates on wikipedia entities."
  python scripts/generate_candidates.py \
      --path_to_model_config data/experiments/${data}/biencoder_${mention_agg_type}/training_params.txt \
      --path_to_model data/experiments/${data}/biencoder_${mention_agg_type}/pytorch_model.bin \
      --entity_dict_path "/private/home/belindali/BLINK/models/entity.jsonl" \
      --encoding_save_file_dir "/private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder" \
      --saved_cand_ids "/private/home/belindali/BLINK/models/entity_token_ids_128.t7" \
      --batch_size 512 \
      --chunk_start ${chunk_start} --chunk_end ${chunk_end}
  python scripts/merge_candidates.py \
      --path_to_saved_chunks /private/home/belindali/BLINK/models/entity_encodings/${data}_${mention_agg_type}_biencoder
fi
