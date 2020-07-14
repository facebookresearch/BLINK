import json

exs = []
with open("saved_preds/webqsp_filtered_dev_pretrain_all_avg_biencoder_true_16_23_joint0/biencoder_outs.jsonl") as f:
    for line in f:
        exs.append(json.loads(line))

print(exs)