# End-to-End Entity Linking


## Data
Data can be found in:
- Entity linking data is under http://dl.fbaipublicfiles.com/elq/EL4QA_data.tar.gz.
- All preprocessed inference data (AIDA-YAGO2/nq/triviaqa/WebQuestions) is under http://dl.fbaipublicfiles.com/elq/all_inference_data.tar.gz
- All preprocessed wikipedia pretraining data is under http://dl.fbaipublicfiles.com/elq/wiki_all_ents.tar.gz
    - WARNING: LARGE!!!
- [FB Internal] Data under `/checkpoint/belindali/entity_link/data/*/tokenized`.

The FAISS indices are under:
- http://dl.fbaipublicfiles.com/elq/faiss_flat_index.pkl
- http://dl.fbaipublicfiles.com/elq/faiss_hnsw_index.pkl
    - Note this differs from HNSW used in BLINK, as it returns inner product score
You can also create your own FAISS indices by running

```console

```

## Model Architecture
**TODO cite paper**
QA-only model is under http://dl.fbaipublicfiles.com/elq/elq_webqsp_only_large.bin
Entity token ids are under http://dl.fbaipublicfiles.com/elq/entity_token_ids_128.t7

## Setup
1. Create conda environment and install requirements
```console
conda create -n el4qa -y python=3.7 && conda activate el4qa
pip install -r requirements.txt
```

2. Download the models and entity embeddings
```console
chmod +x download_models.sh
./download_models.sh
```

## Interactive Mode
```console
export PYTHONPATH=.
python blink/main_dense.py -i --biencoder_model models/elq_wiki_large.bin
```

## Training
### Train on WebQSP
```console
sbatch train_biencoder.sh webqsp all_avg train 128 20 true true large qa_linear
```
Saves under
```
experiments/webqsp/all_avg_20_true_true_bert_large_qa_linear
```

### Train on Wikipedia
```console
sbatch train_biencoder.sh /checkpoint/belindali/entity_link/data/wiki_all_ents all_avg train 32 128 true true large qa_linear 0 -1 22 64
sbatch train_biencoder.sh wiki_all_ents all_avg train 32 128 false false large qa_linear 0 -1 3 64
sbatch train_biencoder.sh wiki_all_ents all_avg train 32 128 false false base qa_linear 0 -1 10 64
```

Saves under
```
experiments/wiki_all_ents/all_avg_128_true_true_bert_large_qa_linear
experiments/wiki_all_ents/all_avg_128_false_false_bert_large_qa_linear
experiments/wiki_all_ents/all_avg_128_false_false_bert_base_qa_linear
```

### Finetune on WebQSP
```console
sbatch train_biencoder.sh webqsp all_avg finetune 32 128 true true large qa_linear 0 -1 0 64 /checkpoint/belindali/entity_link/data/wiki_all_ents ${base_epoch}
```
Saves under
```
experiments/webqsp_ft_wiki_all_ents_${base_epoch}/all_mention_biencoder_all_avg_128_true_true_bert_large_qa_linear
```


## Generating Entity Embeddings
```console
bash get_entity_encodings.sh wiki_all_ents all_avg 128 false false large <epoch_to_pick_up_from>
```
Saves under `experiments/wiki_all_ents/all_avg_128_false_false_bert_large_qa_linear/entity_encodings`

``` console
bash get_entity_encodings.sh wiki_all_ents all_avg 128 false false base <epoch_to_pick_up_from>
```
Saves under `experiments/wiki_all_ents/all_avg_128_false_false_bert_base_qa_linear/entity_encodings`


## Evaluation
Zero-shot from Wikipedia
```console
CUDA_VISIBLE_DEVICES=0 bash run_eval_slurm.sh WebQSP_EL test 'wiki_all_ents;all_avg_128_true_true_bert_large_qa_linear;97' -2.9 50 joint

CUDA_VISIBLE_DEVICES=1 bash run_eval_slurm.sh graphquestions_EL test 'wiki_all_ents;all_avg_128_true_true_bert_large_qa_linear;97' -2.9 50 joint
```

Pretrain on Wikipedia, finetuned on WebQSP
```console
bash run_eval_slurm.sh WebQSP_EL $split 'finetuned_webqsp;all_avg_128_true_true_bert_large_qa_linear;18' -1.5 50 joint

bash run_eval_slurm.sh graphquestions_EL $split 'finetuned_webqsp;all_avg_128_true_true_bert_large_qa_linear;18' -1.5 50 joint
```

Run something on CPUs:
```console
srun --gpus-per-node=0 --partition=learnfair --time=3000 --cpus-per-task 80 --mem=400000 --pty -l bash run_eval_slurm.sh nq ${split} 'finetuned_webqsp;all_mention_biencoder_all_avg_20_true_true_bert_large_qa_linear' -4.5 50 joint 16 false false
```

For Wiki-trained, best threshold is `TODO` (-2.9) for WebQSP, `TODO` (-2.9) for graphquestions, -3.5 for AIDA-YAGO.
For finetuned on WebQSP, best threshold is -1.5 for WebQSP, -0.9 for graphquestions,

Lower thresholds = Predict more candidates = Higher recall/lower precision

The following table summarizes the performance of BLINK for the considered datasets. (Weak matching for WebQSP/GraphQuestions, strong matching for AIDA-YAGO)

model | dataset | biencoder precision | biencoder recall | biencoder F1 | runtime (s), bsz=64, bsz=1 (1CPU), bsz=1 (80CPU) |
------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
WebQSP train | WebQSP test | 0.8999 | 0.8498 | 0.8741 | 183.4 |
Wiki train (e97; HNSW) | WebQSP test | 0.8607 | 0.8181 | 0.8389 | 33.53 |
Pretrain Wiki, Finetune WebQSP | WebQSP test | 0.9170 | 0.8788 | 0.8975 | ? |
Pretrain Wiki, Finetune WebQSP (HNSW index) | WebQSP test | 0.9098 | 0.8704 | 0.8897 | 26.43, 2429.3, 345.2 |
WebQSP train | GraphQuestions test | 0.6010 | 0.5720 | 0.5862 | 756.3 |
Wiki train (e47; HNSW) | GraphQuestions test | 0.6975 | 0.6975 | 0.6975 | 43.32 |
Pretrain Wiki, Finetune WebQSP | GraphQuestions test | 0.7533 | 0.6686 | 0.7084 | ? |
Pretrain Wiki, Finetune WebQSP (HNSW index) | GraphQuestions test | 0.7467 | 0.6641 | 0.7030 | 51.50 |
Wiki train (e23) | AIDA-YAGO2 test(?) | 0.7069 | 0.6952 | 0.7010 | ? |

Timing info for FAISS search vs. biencoder forward run:
(Pretrain Wiki, Fineetune WebQSP on WebQSP test)
* bsz 64 (80 CPUs): forward pass = 10.31s, FAISS search = 0.3123s
* bsz 1 (80 CPUs): forward pass = 0.1636s, FAISS search = 0.0381s 


TODO: make training adversarial selection stricter?


## The BLINK knowledge base
The BLINK knowledge base (entity library) is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format from [http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2)
