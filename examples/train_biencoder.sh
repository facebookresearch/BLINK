#!/bin/sh
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
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

# sbatch examples/train_biencoder.sh pretrain none both 128 <true/false> 0
# sbatch examples/train_biencoder.sh pretrain none predict 512 <true/false> 0
# sbatch examples/train_biencoder.sh webqsp none train 64 false 16

# sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 true true large qa_linear
# sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 true false large qa_linear
#    ^ ADVERSARIAL ABLATE
# sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 false false large qa_linear
#    ^ CAND ENC ABLATE
data=$1  # webqsp_all_ents/graphqs_all_ents/zeshel/pretrain
mention_agg_type=$2  # all_avg/fl_avg/fl_linear/fl_mlp/none/none_no_mentions
objective=$3  # train/predict/both (default)
batch_size=$4  # 128 (for pretraining large model / 128 seqlen) / 32 (for finetuning w/ adversaries / 16 seqlen)
joint_mention_detection=$5  # "true"/false
context_length=$6  # 128/20 (smallest)
load_saved_cand_encs=$7  # true/false
adversarial=$8
model_size=$9  # large/base/medium/small/mini/tiny
mention_scoring_method=${10}  # qa_linear/qa_mlp
chunk_start=${11}
chunk_end=${12}
epoch=${13}


echo $3
export PYTHONPATH=.

# Example to run bi-encoder on zero-shot entity linking data
# Remove --debug flag to run on full dataset
# Set --data_parallel to run it on multiple GPUs
# Increase num_train_epochs to get better models (i.e. 5)

if [ "${data}" = "webqsp" ]
then
  data_path="/private/home/belindali/starsem2018-entity-linking/data/WebQSP"
elif [ "${data}" = "webqsp_all_ents" ]
then
  data_path="/private/home/belindali/starsem2018-entity-linking/data/WebQSP_all_ents"
elif [ "${data}" = "graphqs_all_ents" ]
then
  data_path="/private/home/belindali/starsem2018-entity-linking/data/graphquestions_all_ents"
elif [ "${data}" = "zeshel" ]
then
  data_path="/private/home/ledell/zeshel/data/biencoder/"
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

if [ "${joint_mention_detection}" = "true" ]
then
  echo "doing joint mention detection"
  all_mention_args="${all_mention_args} --do_mention_detection"
fi

cand_enc_args=""
if [ "${load_saved_cand_encs}" = "true" ]
then
  echo "loading + freezing saved candidate encodings"
  cand_enc_args="--freeze_cand_enc --load_cand_enc_only ${cand_enc_args}"
fi

if [ "${adversarial}" = "true" ]
then
  cand_enc_args="--adversarial_training ${cand_enc_args}"
fi

if [ "${objective}" = "" ]
then
  objective="both"
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

if [ "${objective}" = "both" ] || [ "${objective}" = "train" ]
then
  echo "Running ${mention_agg_type} biencoder training on ${data} dataset."
  if [ "${data}" = "pretrain" ]
  then
    if [ "${batch_size}" = "" ]
    then
      batch_size="128"
    fi
    output_path="experiments/pretrain/all_mention_biencoder_${mention_agg_type}_${joint_mention_detection}_${context_length}_${load_saved_cand_encs}_bert_${output_path_model_size}_${mention_scoring_method}"
    if [ "${epoch}" != "-1" ]
    then
      model_path_arg="--path_to_model ${output_path}/epoch_${epoch}/pytorch_model.bin --path_to_trainer_state ${output_path}/epoch_${epoch}/training_state.th"
    fi

    echo "Mention aggregation args: ${all_mention_args}"
    echo "Model path loading args: ${model_path_arg}"
    cmd="python blink/biencoder/train_biencoder.py \
      --output_path ${output_path} \
      --data_path /private/home/ledell/data/wiki_ent2 \
      --num_train_epochs 100 \
      --learning_rate 0.00001 \
      --train_batch_size ${batch_size} \
      --eval_batch_size ${batch_size} \
      --bert_model ${model_ckpt} \
      ${all_mention_args} ${cand_enc_args} \
      --eval_interval 1000 \
      --last_epoch ${epoch} ${model_path_arg} \
      --max_context_length ${context_length} \
      --mention_scoring_method ${mention_scoring_method} \
      --data_parallel"
      # --adversarial_training
      # --debug
      # --start_idx ${chunk_start} --end_idx ${chunk_end}   # TODO DELETE THIS LATER!!!!!
    echo $cmd
    $cmd
  else
    if [ "${batch_size}" = "" ]
    then
      batch_size="32"
    fi
    #--load_cand_enc_only \
    output_path="experiments/${data}/all_mention_biencoder_${mention_agg_type}_${joint_mention_detection}_${context_length}_${load_saved_cand_encs}_bert_${output_path_model_size}_${mention_scoring_method}"
    if [ "${epoch}" != "-1" ]
    then
      model_path_arg="--path_to_model ${output_path}/epoch_${epoch}/pytorch_model.bin --path_to_trainer_state ${output_path}/epoch_${epoch}/training_state.th"
      if [ "${load_saved_cand_encs}" = "true" ]
      then
        cand_enc_args="--freeze_cand_enc --adversarial_training"
      fi
    else
      model_path_arg="--path_to_model /private/home/ledell/BLINK-Internal/models/biencoder_wiki_large.bin"
    fi
    cmd="python blink/biencoder/train_biencoder.py \
      --output_path $output_path \
      ${model_path_arg} --freeze_cand_enc  ${cand_enc_args} \
      --no_cached_representation --dont_distribute_train_samples \
      --data_path ${data_path} \
      --num_train_epochs 100 \
      --learning_rate 0.00001 \
      --max_context_length ${context_length} \
      --max_cand_length 128 \
      --train_batch_size ${batch_size} \
      --eval_batch_size 64 \
      --bert_model ${model_ckpt} \
      --mention_scoring_method ${mention_scoring_method} \
      --eval_interval 500 \
      --last_epoch ${epoch} \
      ${all_mention_args} --data_parallel"  #--debug  #
      # --adversarial_training
    echo $cmd
    $cmd
  fi
fi

# echo "Running ${mention_agg_type} biencoder full evaluation on ${data} dataset."
# python blink/biencoder/eval_biencoder.py \
#   --path_to_model experiments/${data}/biencoder_${mention_agg_type}/pytorch_model.bin \
#   --data_path ${data_path} \
#   --output_path experiments/nn_preds \
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
  
  directory=${data}

  model_config=experiments/${directory}/all_mention_biencoder_${mention_agg_type}_${joint_mention_detection}_${context_length}/training_params.txt
  save_dir=models/entity_encodings/${directory}_${mention_agg_type}_biencoder_${joint_mention_detection}_${context_length}
  if [ "${data}" = "pretrain" ]
  then
    model_path=experiments/${directory}/all_mention_biencoder_${mention_agg_type}_${joint_mention_detection}_${context_length}/epoch_${epoch}/pytorch_model.bin  # TODO REVISE THIS LATER
    save_dir=models/entity_encodings/${directory}_${mention_agg_type}_biencoder_${joint_mention_detection}_${context_length}_${epoch}
  elif [ "${data}" = "zero_shot" ]
  then
    model_path=models/biencoder_wiki_large.bin
    model_config=models/biencoder_wiki_large.json
  else
    model_path=experiments/${directory}/all_mention_biencoder_${mention_agg_type}_${joint_mention_detection}_${context_length}/pytorch_model.bin
  fi
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
  python scripts/generate_candidates.py \
      --path_to_model_config ${model_config} \
      --path_to_model ${model_path} \
      --entity_dict_path "/private/home/belindali/BLINK/models/entity.jsonl" \
      --encoding_save_file_dir "${save_dir}" \
      --saved_cand_ids "/private/home/belindali/BLINK/models/entity_token_ids_128.t7" \
      --batch_size 512 \
      --chunk_start ${chunk_start} --chunk_end ${chunk_end}
  python scripts/merge_candidates.py \
      --path_to_saved_chunks ${save_dir}
fi
