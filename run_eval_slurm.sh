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

# example usage
# bash run_eval_slurm.sh 64 webqsp_filtered zero_shot qa_classifier dev false 0 false
# bash run_eval_slurm.sh 64 webqsp_filtered "finetuned_webqsp;biencoder_none_false_16_2;9 qa_classifier" dev false 0 false
# bash run_eval_slurm.sh 64 webqsp_filtered "finetuned_webqsp;<model_folder>;<best_epoch>" qa_classifier dev false 0 false

# bash run_eval_slurm.sh 64 webqsp_filtered webqsp_none_biencoder qa_classifier dev false 0 false
# bash run_eval_slurm.sh 64 webqsp_filtered zeshel_none_biencoder qa_classifier dev false 0 false
# bash run_eval_slurm.sh 64 webqsp_filtered pretrain_none_biencoder qa_classifier dev false 0 false
# bash run_eval_slurm.sh 64 webqsp_filtered pretrain_all_avg_biencoder qa_classifier dev false 0 false
eval_batch_size=$1  # 64
test_questions=$2  # webqsp_filtered/nq/graphqs_filtered
model_full=$3  # zero_shot/new_zero_shot/finetuned_webqsp/finetuned_graphqs/webqsp_none_biencoder/zeshel_none_biencoder/pretrain_all_avg_biencoder/
ner=$4  # joint/qa_classifier/ngram/single/flair
subset=$5  # test/dev/train_only
debug=$6  # "true"/<anything other than "true"> (does debug_cross)
qa_classifier_threshold=$7  # 0
gpu=$8  # true/false

export PYTHONPATH=.

IFS=';' read -ra MODEL_PARSE <<< "${model_full}"
model=${MODEL_PARSE[0]}
echo $model
echo $model_full

if [ "${test_questions}" = "webqsp_filtered" ]
then
    if [ "${subset}" = "test" ]
    then
        wc="with_classes."
    else
        wc=""
    fi
    # mentions_file=/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.${subset}.entities.${wc}filtered_on_all.json
    mentions_file=/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.${subset}.entities.${wc}all_pos.filtered_on_all.json
    get_predictions="--get_predictions"
elif [ "${test_questions}" = "graphqs_filtered" ]
then
    # mentions_file=/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.${subset}.entities.filtered.json
    mentions_file=/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.${subset}.entities.all_pos.filtered_on_all.no_partials.json
    get_predictions="--get_predictions"
elif [ "${test_questions}" = "nq" ]
then
    mentions_file=/checkpoint/belindali/nq_orig/nq_dev.el.json
    get_predictions="--get_predictions"
fi

if [ "${ner}" = "qa_classifier" ]
then
    qa_classifier_threshold_args="--qa_classifier_threshold ${qa_classifier_threshold}"
else
    qa_classifier_threshold_args=""
fi

if [ "${gpu}" = "false" ]
then
    cuda_args=""
else
    cuda_args="--use_cuda"
fi

if [ "${model}" = "finetuned_webqsp" ]
then
    model_folder=${MODEL_PARSE[1]}  # biencoder_none_false_16_2
    epoch=${MODEL_PARSE[2]}  # 9
    if [[ $epoch != "" ]]
    then
        model_folder=${MODEL_PARSE[1]}/epoch_${epoch}
    fi
    biencoder_config=experiments/webqsp/${MODEL_PARSE[1]}/training_params.txt
    biencoder_model=experiments/webqsp/${model_folder}/pytorch_model.bin
    entity_encoding=/private/home/belindali/BLINK/models/all_entities_large.t7
    crossencoder_config=models/crossencoder_wiki_large.json  # TODO CHANGE!!! (not fintuned yet...)
    crossencoder_model=models/crossencoder_wiki_large.bin  # TODO CHANGE!!! (not fintuned yet...)
# elif [ "${model}" = "finetuned_graphqs" ]
# then
#     entity_encoding=models/all_entities_large.t7
#     biencoder_config=models/biencoder_wiki_large.json
#     biencoder_model=models/biencoder_wiki_large.bin
#     crossencoder_config=models/crossencoder_wiki_large.json
#     crossencoder_model=models/crossencoder_wiki_large.bin
elif [ "${model}" = "zero_shot" ]
then
    entity_encoding=/private/home/belindali/BLINK/models/all_entities_large.t7
    biencoder_config=/private/home/belindali/BLINK/models/biencoder_wiki_large.json
    biencoder_model=/private/home/belindali/BLINK/models/biencoder_wiki_large.bin
    crossencoder_config=/private/home/belindali/BLINK/models/crossencoder_wiki_large.json
    crossencoder_model=/private/home/belindali/BLINK/models/crossencoder_wiki_large.bin
elif [ "${model}" = "new_zero_shot" ]
then
    entity_encoding=models/all_entities_large.t7
    biencoder_config=models/biencoder_wiki_large.json
    biencoder_model=models/biencoder_wiki_large.bin
    crossencoder_config=models/crossencoder_wiki_large.json
    crossencoder_model=models/crossencoder_wiki_large.bin
else
    entity_encoding=models/entity_encodings/${model}/all.t7
    biencoder_config=models/entity_encodings/${model}/training_params.txt
    biencoder_model=models/entity_encodings/${model}/pytorch_model.bin
    crossencoder_config=models/crossencoder_wiki_large.json
    crossencoder_model=models/crossencoder_wiki_large.bin
fi
echo ${mentions_file}

if [ "${debug}" = "true" ]
then
    debug="-dc"
else
    debug=""
fi

command="python blink/main_dense.py -q ${debug} --fast \
    --test_mentions ${mentions_file} \
    --test_entities models/entity.jsonl \
    --entity_catalogue models/entity.jsonl \
    --entity_encoding ${entity_encoding} \
    --biencoder_model ${biencoder_model} \
    --biencoder_config ${biencoder_config} \
    --crossencoder_model ${crossencoder_model} \
    --crossencoder_config ${crossencoder_config} \
    --save_preds_dir saved_preds/${test_questions}_${subset}_${model_full}_${ner}${qa_classifier_threshold} \
    -n ${ner} ${qa_classifier_threshold_args} \
    --eval_batch_size ${eval_batch_size} ${get_predictions} ${cuda_args}"
    # --no_mention_bounds_biencoder --mention_aggregation_type all_avg"
    # --eval_main_entity

echo "${command}"

# python blink/generate_candidate.py  
${command}
