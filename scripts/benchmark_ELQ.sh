# webqsp
bash elq_slurm_scripts/eval_elq.sh WebQSP_EL test finetuned_webqsp -1.5 50 joint 64 false false
bash elq_slurm_scripts/eval_elq.sh WebQSP_EL test wiki_all_ents -2.9 50 joint 64 false false
# graphquestions
bash elq_slurm_scripts/eval_elq.sh graphquestions_EL test finetuned_webqsp -0.9 50 joint 64 false false
bash elq_slurm_scripts/eval_elq.sh graphquestions_EL test wiki_all_ents -2.9 50 joint 64 false false
