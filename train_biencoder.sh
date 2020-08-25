#!/bin/sh
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --partition=priority
#SBATCH --comment=leaving0911
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=USR1
#SBATCH --mem=400000
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time 3000
#SBATCH --constraint=volta32gb

data=$1  # webqsp/graphqs/wiki_all_ents
mention_agg_type=$2  # all_avg/fl_avg/fl_linear/fl_mlp/none/none_no_mentions
objective=$3  # train/predict
batch_size=$4  # 128 (for pretraining large model / 128 seqlen) / 32 (for finetuning w/ adversaries / 16 seqlen)
context_length=$5  # 128/20 (smallest)
load_saved_cand_encs=$6  # true/false
adversarial=$7
model_size=$8  # large/base/medium/small/mini/tiny
mention_scoring_method=$9  # qa_linear/qa_mlp
chunk_start=${10}
chunk_end=${11}
epoch=${12}
eval_batch_size=${13}


echo $3
export PYTHONPATH=.

# Example to run bi-encoder on zero-shot entity linking data
# Remove --debug flag to run on full dataset
# Set --data_parallel to run it on multiple GPUs
# Increase num_train_epochs to get better models (i.e. 5)

if [ "${data}" = "webqsp" ]
then
  data_path="EL4QA_data/WebQSP_EL"
elif [ "${data}" = "graphqs" ]
then
  data_path="EL4QA_data/graphquestions_EL"
elif [ "${data}" = "wiki_all_ents" ]
then
  data_path="/checkpoint/belindali/entity_link/data/tokenized"
fi

if [ "${mention_agg_type}" = "none" ]
then
  all_mention_args=""
elif [ "${mention_agg_type}" = "none_no_mentions" ]
then
  all_mention_args="--no_mention_bounds"
else
  all_mention_args="--no_mention_bounds \
    --mention_aggregation_type ${mention_agg_type}"
fi

cand_enc_args=""
if [ "${load_saved_cand_encs}" = "true" ]
then
  echo "loading + freezing saved candidate encodings"
  cand_enc_args="--freeze_cand_enc --load_cand_enc_only ${cand_enc_args}"
fi
if [ "${adversarial}" = "true" ]
then
  cand_enc_args="--adversarial_training ${cand_enc_args} --index_path models/faiss_hnsw_index.pkl"
fi

if [ "${context_length}" = "" ]
then
  context_length="128"
fi

if [ "${model_size}" = "base" ] || [ "${model_size}" = "large" ]
then
  model_ckpt="bert-${model_size}-uncased"
  output_path_model_size=${model_size}
else
  model_ckpt="/checkpoint/belindali/BERT/${model_size}"
  output_path_model_size=${model_size}
fi

if [ "${epoch}" = "" ]
then
  epoch=-1
fi

if [ "${objective}" = "train" ]
then
  echo "Running ${mention_agg_type} biencoder training on ${data} dataset."
  distribute_train_samples_arg=""
  if [ "${data}" != "wiki_all_ents" ]
  then
    distribute_train_samples_arg="--dont_distribute_train_samples"
  fi

  if [ "${batch_size}" = "" ]
  then
    batch_size="32"
  fi
  if [ "${eval_batch_size}" = "" ]
  then
    eval_batch_size="32"
  fi

  model_path_arg=""
  output_path="experiments/${data}/${mention_agg_type}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${output_path_model_size}_${mention_scoring_method}"
  if [ "${epoch}" != "-1" ]
  then
    model_path_arg="--path_to_model ${output_path}/epoch_${epoch}/pytorch_model.bin --path_to_trainer_state ${output_path}/epoch_${epoch}/training_state.th"
    if [ "${load_saved_cand_encs}" = "true" ]
    then
      cand_enc_args="--freeze_cand_enc --adversarial_training --cand_enc_path models/all_entities_large.t7"
    fi
  else
    if [ "${load_saved_cand_encs}" = "true" ]
    then
      model_path_arg="--path_to_model models/biencoder_wiki_large.bin"
    fi
  fi
  cmd="python blink/biencoder/train_biencoder.py \
    --output_path $output_path \
    ${model_path_arg} ${cand_enc_args} \
    --title_key entity \
    --data_path ${data_path} \
    --num_train_epochs 100 \
    --learning_rate 0.00001 \
    --max_context_length ${context_length} \
    --max_cand_length 128 \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --bert_model ${model_ckpt} \
    --mention_scoring_method ${mention_scoring_method} \
    --eval_interval 500 \
    --last_epoch ${epoch} \
    ${all_mention_args} --data_parallel --get_losses ${distribute_train_samples_arg}"  #--debug  #
  echo $cmd
  $cmd
fi

if [ "${objective}" = "predict" ]
then
  if [ "${chunk_start}" = "" ]
  then
    chunk_start="0"
  fi
  if [ "${chunk_end}" = "" ]
  then
    chunk_end="1000000"
  fi
  
  echo $data

  model_dir=experiments/${data}/${mention_agg_type}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${output_path_model_size}_${mention_scoring_method}

  model_config=${model_dir}/training_params.txt
  model_path=${model_dir}/epoch_${epoch}/pytorch_model.bin
  save_dir=${model_path}/entity_encodings
  mkdir -p save_dir
  chmod 777 save_dir

  if [ ! -f "${save_dir}/training_params.txt" ]
  then
    echo "copying training params from ${model_config} to ${save_dir}/training_params.txt"
    cp ${model_config} ${save_dir}/training_params.txt
  fi

  if [ ! -f "${save_dir}/pytorch_model.bin" ]
  then
    echo "copying saved model bin from ${model_path} to ${save_dir}/pytorch_model.bin"
    cp ${model_path} ${save_dir}/pytorch_model.bin
  fi

  echo "Getting ${mention_agg_type}_${data} biencoder candidates on wikipedia entities."
  cmd="python scripts/generate_candidates.py \
      --path_to_model_config ${model_config} \
      --path_to_model ${model_path} \
      --entity_dict_path models/entity.jsonl \
      --encoding_save_file_dir ${save_dir} \
      --saved_cand_ids models/entity_token_ids_128.t7 \
      --batch_size 512 \
      --chunk_start ${chunk_start} --chunk_end ${chunk_end}"
  echo $cmd
  $cmd
  python scripts/merge_candidates.py \
      --path_to_saved_chunks ${save_dir} \
      --chunk_size $(( chunk_end - chunk_start ))
fi
