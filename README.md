## End-to-End Entity Linking

### Training
```console
sbatch examples/train_biencoder.sh webqsp_all_ents all_avg train 128 true 20 true true large qa_linear
```

```console
sbatch examples/train_biencoder.sh wiki_all_ents all_avg train 32 true 128 false false base qa_linear 0 -1 10 64
```

### Generating Entity Embeddings
```console
bash run_slurm.sh wiki_all_ents all_avg true 128 false false large 2
```

``` console
bash run_slurm.sh wiki_all_ents all_avg true 128 false false base 10
```

### Evaluation
```console
bash run_eval_all_ents_slurm.sh WebQSP_EL test 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear' joint 0.25 100 joint_0
```

The following table summarizes the performance of BLINK for the considered datasets.

| dataset | biencoder precision | biencoder recall | biencoder F1 | runtime (s) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| WebQSP test | 0.8999 | 0.8498 | 0.8741 | 183.4 |


TODO: Release tokenized version of data (alongside original)

## The BLINK knowledge base
The BLINK knowledge base (entity library) is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format from [http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2)
