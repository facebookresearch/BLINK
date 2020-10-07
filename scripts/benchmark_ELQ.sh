# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# wikipedia-trained
echo "Wikipedia-trained, WebQSP eval"
python elq/main_dense.py --threshold=-2.9 \
	--test_mentions EL4QA_data/WebQSP_EL/tokenized/test.jsonl \
	--biencoder_model models/elq_wiki_large.bin \
	--eval_batch_size 64 --max_context_length 32
echo ""
echo "Wikipedia-trained, graphquestions eval"
python elq/main_dense.py --threshold=-2.9 \
	--test_mentions EL4QA_data/graphquestions_EL/tokenized/test.jsonl \
	--biencoder_model models/elq_wiki_large.bin \
	--eval_batch_size 64 --max_context_length 32

# wikipedia-trained, qa finetuned
echo ""
echo "Finetuned on WebQSP, WebQSP eval"
python elq/main_dense.py --threshold=-1.5 \
	--test_mentions EL4QA_data/WebQSP_EL/tokenized/test.jsonl \
	--biencoder_model models/elq_webqsp_large.bin \
	--eval_batch_size 64 --max_context_length 32
echo ""
echo "Finetuned on WebQSP, graphquestions eval"
python elq/main_dense.py --threshold=-0.9 \
	--test_mentions EL4QA_data/graphquestions_EL/tokenized/test.jsonl \
	--biencoder_model models/elq_webqsp_large.bin \
	--eval_batch_size 64 --max_context_length 32

