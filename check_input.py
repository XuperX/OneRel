# to avoid file trunction, check if the input datasets can be processed with tokenizers first.


import transformers
import json
import os

data_dir = "data/CorpusV3"
files = ["train_triples.json", "dev_triples.json", "test_triples.json"]

tokenizer = transformers.BertTokenizer.from_pretrained("pre_trained_bert/scibert_scivocab_cased/vocab.txt")

def checkTokenization(text, tokenizer):
    try:
        tokens = tokenizer.tokenize(text)
    except:
        print("Error tokenizing text: ", text)
        return False


for file in files:
    with open(os.path.join(data_dir, file)) as f:
        print("Checking file: ", file)
        data = json.load(f)
        for item in data:
            checkTokenization(item["text"], tokenizer)

            for triple in item["triple_list"]:
                if len(triple) != 3:
                    print("Error: triple does not have 3 elements: ", triple)
                    continue
                checkTokenization(triple[0], tokenizer)
                checkTokenization(triple[2], tokenizer)
