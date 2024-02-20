"""todo: something is wrong with this script not in use.
this file stores additional configuration files so that I don't need to manually change
rel_num, bert_model time every time I switch between models.
"""

import os
import json

bert_model_name = None
bert_model_dir = None
rel_num = None
def initialize_config(model_name, model_dir,dataset):
    global bert_model_name, bert_model_dir, rel_num
    bert_model_name = model_name
    bert_model_dir = os.path.join("./pre_trained_bert", model_dir)
    rel_json_file = os.path.join("data", dataset, "rel2id.json")
    with open(rel_json_file, "r") as f:
        rel2id = json.load(f)
    rel_num = len(rel2id[0])

print(f"Initialized bert_model_dir: {bert_model_dir}")
print(f"Using bert_model_dir: {bert_model_dir}")

