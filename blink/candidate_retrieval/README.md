# [deprecated] BLINK with Solr Canditate Retrieval

For _candidate retrieval_ the [Solr](https://lucene.apache.org/solr) search engine is used.
BLINK provides scripts to download, process and link the lasest wikipedia and wikidata dumps, ingest the resulting 5.9M entities into the search engine index, as well as scripts to evaluate the retrieval phase on well known benchmarking data.

For _candidate ranking_ BLINK can use a model based on _BERT-large-cased_ trained on the [AIDA corpus](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida) with the [pytorch-transformers](https://github.com/huggingface/transformers) library. Scripts to train and evaluate the ranking model are provided.

These three components could be extended and upgraded independently. 
On the retrieval part, the candidate catalog can be replaced or extended, and the retrieval model upgraded.
On the ranking side, the BERT model could be replaced with other pre-trained language models (such as RoBERTa) and improved.

Finally, BLINK provides a script to annotate and link all the entities in a given document.

## Install BLINK

It might be a good idea to use a separate conda environment. It can be created by running:

```
conda create -n blink37 -y python=3.7 && conda activate blink37
```

```
git clone git@github.com:facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
export PYTHONPATH=.
```


Below are the instructions to setup a [Solr](https://lucene.apache.org/solr) instance for the retrieval, and train a BERT based model for the ranking component.

### 1. Training and benchmarking data

First things first. Download the training (AIDA-train), development (AIDA-A) and benchmarking (AIDA-B, ACE2004, AQUAINT, WNED-CWEB, MSNBC, WNED-WIKI) data

```
chmod +x get_train_and_benchmark_data.sh
./get_train_and_benchmark_data.sh
```

### 2. Candidate Retrieval

Follow the instructions on the [offical website](https://lucene.apache.org/solr) to install the Solr search platform. BLINK is currently running with version 8.2.0.

Download and process the data for the knowledge base

```
chmod +x blink/candidate_retrieval/scripts/get_processed_data.sh
./blink/candidate_retrieval/scripts/get_processed_data.sh data
```

Index the knowledge base
```
chmod +x blink/candidate_retrieval/scripts/ingestion_wrapper.sh
./blink/candidate_retrieval/scripts/ingestion_wrapper.sh data
```
this will create a Solr collection named wikipedia, that you can access through a GUI [here](http://localhost:8983/solr/#/wikipedia/core-overview)

Generate candidates for all of the mentions in the training and benchmarking data
```
python blink/candidate_retrieval/perform_and_evaluate_candidate_retrieval_multithreaded.py \
--num_threads 70 \
--dump_mentions \
--dump_file_id train_and_eval_data \
--include_aida_train
```
and save them. The previous also evaluates the candidate retrieval component.

Finally process the saved output to get a JSON format representation of each dataset, which would be used by the ranking component.

```
python blink/candidate_retrieval/json_data_generation.py
```

### 3. Candidate Ranking

With the data in place, we now train and evaluate a BERT based ranker
```
python blink/candidate_ranking/train.py \
--model_output_path models/bert_large_ranking \
--evaluate --full_evaluation --dataparallel_bert
```
In order to facilitate the training with 80 candidates with BERT-large, using 512 tokens representations, 8 x 32 Gb GPUs are needed. Alternatively, the requirements can be substatially alleviated by varying any of the aforementioned parameters. 

To only evaluate an already trained model use the following command
```
python blink/candidate_ranking/evaluate.py \
--path_to_model models/bert_large_ranking \
--full_evaluation \
--dataparallel_bert \
--evaluation_batch_size 1
```

## Use BLINK
Once the Solr instance has been setup and a trained model is available, you can use blink in the following way

```
python blink/main_solr.py \
  --i examples/text.txt \
  --m models/bert_large_ranking \
  --o output/
```

The API assumes that the input is a document, so it will first split the document on its constituent sentences (alternativelly, the input can be provided as once sentence per line), identify the mentions and link them to entities from the knowledge base.
