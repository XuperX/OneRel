An adaptation of `taishan1994/OneRel_chinese`, summarised from the discussions in the issue section.

## Changes:
1. revised the data loader to ensure if a sentence triple is not correctly processed, it will be skipped.

## Common issues:
1. If the F1 is 0.000, train more epoch. Depending on the size of the dataset. Sometimes, it takes  more than 100 epoch training before seeing performance improvements.

## Run this code.
This code is semi-automatic. There are several things to change before executing.
1. Add your model to the `pre_trained_bert` folder. The pretrained model can be downloaded from huggingface.
	
 For example, if using "scibert_scivocab_cased" as the pre-defined bert model. Then the file structure should be `pre_trained_bert/scibert_scivocab_cased`. There should be at least three files, including the bin file, config.json, vocab.txt, in the directory.
 
2. change the model path in `models/rel_model.py`
   e.g. 
   > self.bert_encoder = BertModel.from_pretrained('./pre_trained_bert/scibert_scivocab_cased', local_files_only=True) 
   
   Note if run in internal server which do not have access to the internet, specify local_files_only.
   
4. change the model vocabulary path in `data_loader.py`
   e.g.
   > tokenizer = get_tokenizer('pre_trained_bert/scibert_scivocab_cased/vocab.txt')
   
5. Put data in the right directory. e.g. `data/datasetX`

   DatasetX should contain `train/dev/test_triples.json`, as well as a `rel2id.json`
   
7. change the number of relationships in `data_loader.py`
   e.g.
   > batch_triple_matrix = torch.LongTensor(cur_batch_len, 16, max_text_len, max_text_len).zero_()
   #"16" is the number of total relationships, which can be found in the `data/datasetX/rel2id.json` file.
   
5. create a new directory for checkpoints.
   e.g. `checkpount/datasetX`
   
5. Execution
   ** The `jobscript_colab` file can be executed automatically. ** Though it is not a very clean script.

   > python train.py --dataset=corpus3  --batch_size=2 --max_epoch=500 --rel_num=16 --max_len=1500 
   > #if `--batch_size>=2" it might require more than 16GB memory. 
   > #`--max_epoch` at least 100 for a regular sized (number of train sentences ~ 500)  dataset to ensure there are valid outputs.
## If run on an internal server.
   1. ensure the model is loaded from local files by doing:
   - revise the model loading script
   	> self.bert_encoder = BertModel.from_pretrained('./pre_trained_bert/scibert_scivocab_cased', local_files_only=True)
   
   -  ensure the script is executed from a local environment
      > export TF_ENABLE_ONEDNN_OPTS=0
        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 # Adjust to manage memory fragmentation
        python train.py --dataset=corpus3  --batch_size=2 --max_epoch=500 --rel_num=16 --max_len=1500
      
    2. if batch_size > 2, make sure you have enough memory, and use `--multi-gpu` when needed.
    
    3. given not all servers have the most up-to-date software, and the compatiablity issues. Below is a set of configurations which are old but executable. 
       ```
       torch                     1.11.0                   pypi_0    pypi
       tqdm                      4.66.1                   pypi_0    pypi
       transformers              4.5.1                    pypi_0    pypi
       tensorflow                2.8.0                    pypi_0    pypi
       keras                     2.8.0                    pypi_0    pypi 
       keras-bert                0.88.0                   pypi_0    pypi
       ```



## Below are from the original project description
 OneRel: Joint Entity and Relation Extraction with One Model in One Step

This repository contians the source code and datasets for the paper: **OneRel: Joint Entity and Relation Extraction with One Model in One Step**, Yu-Ming Shang, Heyan Huang and Xian-Ling Mao, AAAI-2022.

## Motivation

Most existing joint entity and relaiton extraction methods suffer from the problems of cascading errors and redundant information. We think that the fundamental reason for the problems is that the decomposition-based paradigm ignores an important property of a triple -- its head entity, relation and tail entity are interdependent and indivisible. In other words, it is unreliable to extract one element without fully perceiving the information of the other two elements. Therefore, this paper propose a novel perspective for joint entity and relation extraction, that is, transforming the task into a triple classification problem, making it possible to capture the information of head entities, relations and tail entities at the same time.

## Relation-Specific Horns Tagging

In order to decode entities and relations from the output matrix accurately and efficiently, we introduce a relation-specific horns tagging, as shown in the figure. So, for each relation, the spans of head entities are spliced from "HB-TE" to "HE-TE"; the spans of tail entities are spliced from "HB-TB" to "HB-TE"; and two paired entities share the same "HB-TE".

![tagging](/img/tagging.png)

## Usage

1. **Environment**
   ```shell
   conda create -n your_env_name python=3.8
   conda activate your_env_name
   cd OneRel
   pip install -r requirements.txt
   ```

2. **The pre-trained BERT**

    The pre-trained BERT (bert-base-cased) will be downloaded automatically after running the code. Also, you can manually download the pre-trained BERT model and decompress it under `./pre_trained_bert`.


3. **Train the model (take NYT as an example)**

    Modify the second dim of `batch_triple_matrix` in `data_loader.py` to the number of relations, and run

    ```shell
    python train.py --dataset=NYT --batch_size=8 --rel_num=24 
    ```
    The model weights with best performance on dev dataset will be stored in `checkpoint/NYT/`

4. **Evaluate on the test set (take NYT as an example)**

    Modify the `model_name` (line 48) to the name of the saved model, and run 
    ```shell
    python test.py --dataset=NYT --rel_num=24
    ```

    The extracted results will be save in `result/NYT`.

## Results
To reproduce the performance of the paper, please download our model states [here](https://drive.google.com/drive/folders/1VKd0Y3kSXQ8Vf8W7ZudEUEn2FNx6nSLr?usp=sharing).


## **Acknowledgment**
I 
I followed the previous work of [longlongman](https://github.com/longlongman/CasRel-pytorch-reimplement) and [weizhepei](https://github.com/weizhepei/CasRel). 

So, I would like to express my sincere thanks to them. 



