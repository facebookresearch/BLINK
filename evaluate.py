# This Python file uses the following encoding: utf-8
import blink.main_dense as main_dense
import argparse
import json
import blink.candidate_ranking.utils as utils

biencoder_models_path = "./result/biencoder/"  # the path where you stored the BLINK models
crossencoder_models_path = "./result/crossencoder/"  # the path where you stored the BLINK models
result_file = "./fasih/result/result.txt"
data_path = "DATA_PATH/blink_format/"
output_path = "./result/output_prediction-50"
logger = utils.get_logger(output_path)
config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 64,
    "biencoder_model": biencoder_models_path + "pytorch_model.bin",
    "biencoder_config": biencoder_models_path + "training_params.txt",
    "entity_catalogue": "DATA_PATH/documents/wikipedia.jsonl",
    "entity_encoding": "./result/eval/encode.t7",
    "crossencoder_model": crossencoder_models_path + "pytorch_model.bin",
    "crossencoder_config": crossencoder_models_path + "training_params.txt",
    "fast": False,  # set this to be true if speed is a concern
    "output_path": output_path  # logging directory
}

args = argparse.Namespace(**config)
models = main_dense.load_models(args, logger=logger)
print('model loaded...')

with open(result_file, "w") as file1:
    file1.write("")


def log_result(name):
    config["test_mentions"] = data_path + name + '.jsonl'
    args = argparse.Namespace(**config)
    biencoder_accuracy, recall_at, \
        crossencoder_normalized_accuracy, \
        overall_unormalized_accuracy, num_datapoints, \
        predictions, scores = main_dense.run(
        args, logger, *models)
    with open(result_file, "a", encoding='utf-8') as file1:
        # Writing data to a file
        res_text = f'Mode: {name}, ' \
                   f'biencoder_accuracy: {round(biencoder_accuracy, 2)}, recall_at: {round(recall_at, 2)} , ' \
                   f'crossencoder_normalized_accuracy: {round(crossencoder_normalized_accuracy, 2)}, ' \
                   f'overall_unormalized_accuracy : {round(overall_unormalized_accuracy, 2)},' \
                   f'support : {num_datapoints}\n----------------------------\n '
        print(res_text)
        file1.write(res_text)
    with open(f'./result/{name}_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f)


# Test
log_result('test')

# Train
log_result('train')

# Valid
log_result('valid')
