## End-to-End Entity Linking

All data is under `/checkpoint/belindali/entity_link/data/*/tokenized`

**TODO: Release tokenized version of data (alongside original)**

### Training
```console
sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 true true large qa_linear
```

```console
sbatch examples/train_biencoder.sh wiki_all_ents all_avg train 32 true 128 true true large qa_linear 0 -1 11 64
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
Saves under
```
models/entity_encodings/wiki_all_ents_all_avg_true_128_false_false_bert_large_qa_linear
```

``` console
bash run_slurm.sh wiki_all_ents all_avg true 128 false false base <epoch_to_pick_up_from>
```
Saves under
```
models/entity_encodings/wiki_all_ents_all_avg_true_128_false_false_bert_base_qa_linear
```

### Evaluation
```console
bash run_eval_all_ents_slurm.sh WebQSP_EL test 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear' joint 0.25 100 joint_0
```

The following table summarizes the performance of BLINK for the considered datasets.

| dataset | biencoder precision | biencoder recall | biencoder F1 | runtime (s), bsz=64 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| WebQSP test | 0.8999 | 0.8498 | 0.8741 | 183.4 |



## The BLINK knowledge base
The BLINK knowledge base (entity library) is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format from [http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2)
