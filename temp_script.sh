for split in test2 test5 test8 test11
do
    bash run_eval_all_ents_slurm.sh triviaqa $split 'finetuned_webqsp_all_ents;all_mention_biencoder_all_avg_true_20_true_true_bert_large_qa_linear' joint 0.25 100 joint_0 16
done