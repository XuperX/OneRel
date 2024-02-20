## cehck if input files are valid or not.
"""
1. all relationships have been included in the rel2id.json file.
2. all tags have "text" > 0, and "triple_list" > 0.
"""

import json
import os

dir = "data/corpus3"
files = ["train_triples.json", "dev_triples.json", "test_triples.json"]
rel2id = json.load(open(os.path.join(dir, "rel2id.json"), "r"))

for file in files:
    with open(os.path.join(dir, file), "r") as f:
        data = json.load(f)

        # check if all relationships are included in rel2id.json
        for item in data:
            for triple in item["triple_list"]:
                if triple[1] not in rel2id[1]:
                    print(f"Error: {triple[1]} not in rel2id.json")
            if len(item["text"]) == 0:
                print(f"Error: empty text in {file}")
            if len(item["triple_list"]) == 0:
                print(f"Error: empty triple_list in {file}")

        # check if all tags have "text" > 0 and the max_len is reasonale
            max_text_len = 0
            min_text_len = 1
            for item in data:
                if len(item["text"]) > max_text_len:
                    max_text_len = len(item["text"])
                if len(item["text"]) <= min_text_len:
                    print(item)
                    min_text_len = len(item["text"])
        print(f"Max text length: {max_text_len}")
        print(f"Min text length: {min_text_len}")