## End-to-End Entity Linking

All data is under `/checkpoint/belindali/entity_link/data/*/tokenized`. The public link is https://dl.fbaipublicfiles.com/elq/EL4QA_data.tar.gz.

The FAISS indices are under:
- https://dl.fbaipublicfiles.com/elq/faiss_flat_index.pkl
- https://dl.fbaipublicfiles.com/elq/faiss_hnsw_index.pkl
    - Note this differs from HNSW used in BLINK, as it returns inner product score
You can also create your own FAISS indices by running
```console

```

**TODO: Release tokenized version of data (alongside original)**
### Model Architecture
**TODO cite paper**

### Setup
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

### Interactive Mode
```console
python blink/main_dense_all_ents.py -i \
    --test_entities models/entity.jsonl \
    --entity_catalogue models/entity.jsonl \
    --entity_encoding models/all_entities_large.t7 \
    --biencoder_model experiments/wiki_all_ents/all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear/epoch_22/pytorch_model.bin \
    --biencoder_config experiments/wiki_all_ents/all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear/training_params.txt \
    --threshold -4.5 --num_cand_mentions 50 --num_cand_entities 10 \
    --threshold_type joint --faiss_index hnsw --index_path models/faiss_hnsw_index.pkl
```

### Training
Train on WebQSP
```console
sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 true true large qa_linear
```
Saves under
```
experiments/webqsp_all_ents/all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear
```

Finetune on WebQSP
1. Copy pretraining checkpoint directory `experiments/wiki_all_ents/*/epoch_*` to `experiments/webqsp_all_ents/all_mention_biencoder_all_avg_true_32_true_true_bert_large_qa_linear/epoch_0`
2. Delete the saved trainer state (to reset trainer from scratch): `rm experiments/webqsp_all_ents/all_mention_biencoder_all_avg_true_32_true_true_bert_large_qa_linear/epoch_0/training_state.th`
3. Run:
```console
sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 32 true 128 true true large qa_linear 0 -1 0
```
Saves under
```
experiments/webqsp_all_ents/all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear
```

Train on Wikipedia
```console
sbatch examples/train_biencoder.sh wiki_all_ents all_avg train 32 true 128 true true large qa_linear 0 -1 22 64
sbatch examples/train_biencoder.sh wiki_all_ents all_avg train 32 true 128 false false large qa_linear 0 -1 3 64
sbatch examples/train_biencoder.sh wiki_all_ents all_avg train 32 true 128 false false base qa_linear 0 -1 10 64
```

Saves under
```
experiments/wiki_all_ents/all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear
experiments/wiki_all_ents/all_mention_biencoder_all_avg_true_128_false_false_bert_large_qa_linear
experiments/wiki_all_ents/all_mention_biencoder_all_avg_true_128_false_false_bert_base_qa_linear
```


### Generating Entity Embeddings
```console
bash run_slurm.sh wiki_all_ents all_avg true 128 false false large <epoch_to_pick_up_from>
```
Saves under `models/entity_encodings/wiki_all_ents_all_avg_true_128_false_false_bert_large_qa_linear`

``` console
bash run_slurm.sh wiki_all_ents all_avg true 128 false false base <epoch_to_pick_up_from>
```
Saves under `models/entity_encodings/wiki_all_ents_all_avg_true_128_false_false_bert_base_qa_linear`


### Evaluation
Zero-shot from Wikipedia
```console
CUDA_VISIBLE_DEVICES=0 bash run_eval_all_ents_slurm.sh WebQSP_EL test 'wiki_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;49' -2.9 50 joint

CUDA_VISIBLE_DEVICES=1 bash run_eval_all_ents_slurm.sh graphquestions_EL test 'wiki_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;49' -2.9 50 joint
```

Pretrain on Wikipedia, finetuned on WebQSP
```console
bash run_eval_all_ents_slurm.sh WebQSP_EL $split 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;18' -1.5 50 joint

bash run_eval_all_ents_slurm.sh graphquestions_EL $split 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_128_true_true_bert_large_qa_linear;18' -1.5 50 joint
```

Run something on CPUs:
```console
srun --gpus-per-node=0 --partition=learnfair --time=3000 --cpus-per-task 80 --mem=400000 --pty -l bash run_eval_all_ents_slurm.sh nq ${split} 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear' -4.5 50 joint 16
```

For Wiki-trained, best threshold is `TODO` (-2.9) for WebQSP, `TODO` (-2.9) for graphquestions, -3.5 for AIDA-YAGO.
For finetuned on WebQSP, best threshold is -1.5 for WebQSP, -0.9 for graphquestions,

Lower thresholds = Predict more candidates = Higher recall/lower precision

The following table summarizes the performance of BLINK for the considered datasets. (Weak matching for WebQSP/GraphQuestions, strong matching for AIDA-YAGO)

model | dataset | biencoder precision | biencoder recall | biencoder F1 | runtime (s), bsz=64, bsz=1 (1CPU), bsz=1 (80CPU) |
------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
WebQSP train | WebQSP test | 0.8999 | 0.8498 | 0.8741 | 183.4 |
Wiki train (e49; HNSW) | WebQSP test | 0.8607 | 0.8181 | 0.8389 | 33.53 |
Pretrain Wiki, Finetune WebQSP | WebQSP test | 0.9170 | 0.8788 | 0.8975 | ? |
Pretrain Wiki, Finetune WebQSP (HNSW index) | WebQSP test | 0.9098 | 0.8704 | 0.8897 | 26.43, 2429.3, 328.1 |
WebQSP train | GraphQuestions test | 0.6010 | 0.5720 | 0.5862 | 756.3 |
Wiki train (e49; HNSW) | GraphQuestions test | 0.6975 | 0.6975 | 0.6975 | 43.32 |
Pretrain Wiki, Finetune WebQSP | GraphQuestions test | 0.7533 | 0.6686 | 0.7084 | ? |
Pretrain Wiki, Finetune WebQSP (HNSW index) | GraphQuestions test | 0.7450 | 0.6555 | 0.6974 | 52.12 |
Wiki train (e23) | AIDA-YAGO2 test(?) | 0.7069 | 0.6952 | 0.7010 | ? |

TODO: make training adversarial selection stricter?


## The BLINK knowledge base
The BLINK knowledge base (entity library) is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format from [http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2)
