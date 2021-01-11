## Example to train BLINK on Zero-shot Entity Linking dataset

Download dataset:

    ./examples/zeshel/get_zeshel_data.sh
 
Convert data to BLINK format:

    python examples/zeshel/create_BLINK_zeshel_data.py

Train Biencoder model. Note: the following command requires to run on 8 GPUs with 32G memory. Reduce the train_batch_size and eval_batch_size for less GPUs/memory resources.

    python blink/biencoder/train_biencoder.py \
      --data_path data/zeshel/blink_format \
      --output_path models/zeshel/biencoder \  
      --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 \
      --train_batch_size 128 --eval_batch_size 64 --bert_model bert-large-uncased \
      --type_optimization all_encoder_layers --data_parallel

Get top-64 predictions from Biencoder model on train, valid and test dataset:

    python blink/biencoder/eval_biencoder.py \
      --path_to_model models/zeshel/biencoder/pytorch_model.bin \
      --data_path data/zeshel/blink_format \
      --output_path models/zeshel \
      --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result \
      --bert_model bert-large-uncased --mode train,valid,test \
      --zeshel True --data_parallel

Train and eval crossencoder model:

    python blink/crossencoder/train_cross.py \
      --data_path  models/zeshel/top64_candidates/ \
      --output_path models/zeshel/crossencoder \
      --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 \
      --train_batch_size 2 --eval_batch_size 2 --bert_model bert-large-uncased \
      --type_optimization all_encoder_layers --add_linear --data_parallel \
      --zeshel True
