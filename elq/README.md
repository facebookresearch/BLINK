# ELQ

ELQ (<ins>E</ins>ntity <ins>L</ins>inking for <ins>Q</ins>uestions) is an python library that links entities in questions to the Wikipedia knowledge base.

***NOTE: THE FOLLOWING COMMANDS ARE ALL INTENDED TO BE RUN FROM THE PARENT DIRECTORY (i.e. `BLINK` rather than `BLINK/elq`)***


## ELQ architecture

The ELQ architecture is described in the following paper:

```bibtex
@inproceedings{li2020efficient,
 title={ Efficient One-Pass End-to-End Entity Linking for Questions },
 author={ Li, Belinda Z. and Min, Sewon and Iyer, Srinivasan and Mehdad, Yashar and Yih, Wen-tau },
 booktitle={ EMNLP },
 year={2020}
}
```

[https://arxiv.org/pdf/2010.02413.pdf](https://arxiv.org/pdf/2010.02413.pdf)

## Data
The question entity linking data is under http://dl.fbaipublicfiles.com/elq/EL4QA_data.tar.gz


## Setup
1. Create conda environment and install requirements (this step can be skipped if you've already created an environment for BLINK)
```console
conda create -n el4qa -y python=3.7 && conda activate el4qa
pip install -r elq/requirements.txt
```

2. Download the pretrained models, indices, and entity embeddings
```console
chmod +x download_elq_models.sh
./download_elq_models.sh
```

To download the flat (exact search) indexer, you may use the same flat index as BLINK: [BLINK flat index](http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl)

To build and save FAISS (exact search) index yourself, run
`python blink/build_faiss_index.py --output_path models/faiss_flat_index.pkl`


## Interactive Mode
To run ELQ in interactive mode, run the following command:
```console
export PYTHONPATH=.
python elq/main_dense.py -i
```
The model takes around 1.5 minutes to start up, after which each question should take at most a second.

By default, we use the Wikipedia-trained model. You may use the model finetuned on questions data by specifying `--biencoder_model models/elq_webqsp_large.bin`.

You may also optionally change the threshold for each question, which allows you to adjust how many entities the model outputs. By default the threshold is `-4.5`. 
Higher threholds decreases the number of output candidates (up to `0.0`). Lower thresholds increases the number of output candidates.

By default, ELQ uses approximate search with a FAISS hnsw index. For exact search, you can:
1. run ELQ without an index: by specifying `--faiss_index none`
2. run ELQ with an exact flat index: by specifying `--faiss_index flat --index_path models/faiss_flat_index.pkl`


## Use ELQ in your codebase

```console
pip install -e git+git@github.com:facebookresearch/BLINK#egg=BLINK
```

```python
import elq.main_dense as main_dense
import argparse

models_path = "models/" # the path where you stored the ELQ models

config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link = [{
                    "id": 0,
                    "text": "paris is capital of which country?".lower(),
                },
                {
                    "id": 1,
                    "text": "paris is great granddaughter of whom?".lower(),
                },
                {
                    "id": 2,
                    "text": "who discovered o in the periodic table?".lower(),
                },
                ]

predictions = main_dense.run(args, None, *models, test_data=data_to_link)
```

## Benchmarking ELQ

To benchmark ELQ against Question Entity Linking datasets (`WebQSP_EL` and `graphquestions_EL`), run the following command:

```console
bash scripts/benchmark_ELQ.sh  
```

The following table summarizes the performance of ELQ for the considered datasets.

model | dataset | threshold | weak-match precision | weak-match recall | weak-match F1 | EM precision | EM recall | EM F1 | 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
Wikipedia-trained | WebQSP_EL test | -2.9 | 0.8607 | 0.8181 | 0.8389 | 0.8607 | 0.7975 | 0.7581 | 0.7773 |
Wikipedia-train | graphquestions_EL test | -2.9 | 0.6975 | 0.6975 | 0.6975 | 0.6212 | 0.6212 | 0.6212 |
Finetuned on WebQSP | WebQSP_EL test | -1.5 | 0.9109 | 0.8815 | 0.8960 | 0.8741 | 0.8459 | 0.8598 |
Finetune on WebQSP | graphquestions_EL test | -0.9 | 0.7394 | 0.6700 | 0.7030 | 0.6716 | 0.6086 | 0.6386 |

## Train ELQ

We provide scripts for training ELQ on your own data.
```console
bash scripts/train_ELQ.sh \
  train <train_data_path> <max_context_len> \
  <train_batch_size> <eval_batch_size>
```
which saves model checkpoints under
```
experiments/<path_to_data>/all_avg_<max_context_len>_true_true_bert_large_qa_linear
```
### Finetuning
We also allow you to finetune a pretrained model (i.e. trained from Wikipedia).
```console
bash scripts/train_ELQ.sh \
  finetune <train_data_path> <max_context_len> \
  <train_batch_size> <eval_batch_size> 0 \
  <base_training_data> <base_epoch>
```

### Examples
For example, training on WebQSP_EL from scratch is run with:
```console
bash scripts/train_ELQ.sh train EL4QA_data/WebQSP_EL/tokenized 20 128 64
```
and saves under
```
experiments/webqsp/all_avg_20_true_true_bert_large_qa_linear
```

Finetuning on WebQSP_EL, from (epoch 97 of) a Wikipedia-trained model looks like:
```console
bash scripts/train_ELQ.sh finetune EL4QA_data/WebQSP_EL/tokenized 20 128 64 0 wiki_all_ents 97
```
and saves under
```
experiments/webqsp_ft_wiki_all_ents_97/all_mention_biencoder_all_avg_128_true_true_bert_large_qa_linear
```


## The ELQ knowledge base
The ELQ knowledge base (entity library) is identical to BLINK, based on the 2019/08/01 Wikipedia dump.


## Troubleshooting

If the module cannot be found, preface the python command with `PYTHONPATH=.`

