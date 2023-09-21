## Setup Environment

```shell
conda create -n blink37 -y python=3.7 && conda activate blink37
pip install -r requirements.txt --default-timeout=100
```

## Train bi-encoder

Set the Python path and train the bi-encoder

```shell
PYTHONPATH=. python3 blink/biencoder/train_biencoder.py   --data_path DATA_PATH \
    --output_path ./result/biencoder   --bert_model BERT_MODEL --num_train_epochs 25 \
    --eval_batch_size 64   --train_batch_size 64  \
    --type_optimization all_encoder_layers --data_parallel  --print_interval  100 \
    --eval_interval 2000  --save_topk_result --mode Train --top_k 10
```

## Get top-k predictions

Get top-k predictions using the bi-encoder

```shell
PYTHONPATH=. python blink/biencoder/eval_biencoder.py --data_path DATA_PATH \
--save_topk_result  --path_to_model ./result/biencoder/pytorch_model.bin \
--mode train,valid,test --eval_batch_size 16 --output_path ./result/top_k \
--max_context_length 128 --max_cand_length 128 --bert_model BERT_MODEL \
--entity_dict_path DATA_PATH/documents/wikipedia.jsonl \
--cand_pool_path ./result/eval/pool.t7 --cand_encode_path ./result/eval/encode.t7 --top_k 64
```

## Train cross-encoder

Train the cross-encoder

```shell
PYTHONPATH=. python blink/crossencoder/train_cross.py   --data_path  ./result/top_k/top64_candidates/ \
--output_path ./resul/crossencoder    --num_train_epochs 5 --max_context_length 128 \
--max_cand_length 128   --train_batch_size 1 --eval_batch_size 1 --bert_model BERT_MODEL \
--type_optimization all_encoder_layers --data_parallel --print_interval  400 \
--eval_interval 5000 --mode Train --add_linear --top_k 64 --evaluate
```

## Evaluate

Consider any adjustments that could be required in evaluate.py.

```shell
PYTHONPATH=. python evaluate.py
```

A Huggingface model is BERT_MODEL, for instance: "bert-base-uncased"