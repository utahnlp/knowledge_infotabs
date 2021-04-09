# Incorporating External Knowledge to Enhance Tabular Reasoning

<p align="center">
    <img width="18%" src="logo_knowledge.jpg" /> 
    <img width="70%" src="logo_infotabs.png" />
</p>

Implementation of the knowledge incorporation for semi-structured inference (InfoTabS) our [NAACL 2021](https://2021.naacl.org/) paper: [Incorporating External Knowledge to Enhance Tabular Reasoning](https://vgupta123.github.io/docs/Knowledge_InfoTabS_Camera_Ready.pdf). 

## 0. Prerequisites
Download and unpack INFOTABS dataset into ```./data``` in the main code folder. This step is optional as we have already provided the INFOTABS data which can be found in ```/data/maindata/``` and ```/data/tables/ ```


```
conda env create --name myenv --file environment.yaml
conda activate myenv
```

some packages in the conda environment are required to be installed by pip

```
pip install -r requirements.txt
```

## 1. KG data (optional)
We are using bert-as-service which utlise cloud TPU for disambiguation purpose. Look up [here](https://github.com/hanxiao/bert-as-service) for the documentation. For convenience we have already provided the extracted kg data which can be found in ```/data/kgdata/```

```

cd scripts
cd kg_extraction

```

However you can extract kg as follows:


### 1.1 Download pre-trained BERT Model:

```
mkdir -p tmp
```

Download the BERT-Base, Uncased model, from [here](https://github.com/hanxiao/bert-as-service) and uncompress the zip file into the folder ```tmp```. 

### 1.2 Start the BERT service:
```uncased_L-12_H-768_A-12``` is the name of the BERT folder downloaded.

```

bert-serving-start -model_dir tmp/uncased_L-12_H-768_A-12/ -max_seq_len NONE -pooling_strategy NONE -show_tokens_to_client 

```

### 1.3 KG Extraction:
We can extract the KG data by running the following line of code in a different terminal to the server:

```
python3 extract_kg.py 
```

The data is extracted into ```/data/kgdata/```

## 2. Preprocessing (optional)
```data/maindata``` and ```data/tables``` will be the primary folders we use for the preprocessing

### 2.1 Preprocessing
Preprocessing is divided into two steps

First generate the premises for the steps BPR, DRR and KG explicit from the json files. Assume the data is downloaded and unpacked into ```data/maindata/```.  For convenience we have already provided the preprocessed files in ```/temp/data/``` . However you can also generate them as follows:
 
There are four ```.py``` files for preprocessing ```bpr.py, opr.py, drr.py, kg_explicit.py```

```bpr.py``` : It is used to generate data for the Better Paragraph Representation (BPR) step. We also use this file to generate better paragraph representation but with a newline character in between sentences to split the sentences better for the DRR or Distracting Row Removal step.

```

important argument details in bpr.py

--json_dir set as the directory path of the json files, i.e we set it to a folder named ./../data/tables/json/ in this case
--mode set as 1 to add newline character between every sentence (this makes it easier to split sentences while performing drr step as splitting by . splits abbreviations and leads to unfavourale situations). Set to 0 for normal paragraph used in bpr. Default value is set to 0.
--map set to "mnli" for mapping C-0, N-1 and E-2 (we use this as while using implicit knowledge the model is pretrained considering this mapping and hence it makes things easier). We set this to default as opposed to the mapping E-0, N-1 and C-2 used in the InfoTabS code.
--data_dir set to the directory containing the maindata and the hypothesis etc. i.e. we set it to a folder named ./../data/maindata/ in this case
--save_dir set to the directory where the generated bpr data will be saved. Here we have set it to ./../../temp/data/bpr/
--cat_dir set to the directory containing the table_categories.tsv file with the categories of each table
--splits data split names for which the bpr is to be generated. Default is set to all the splits ["train","dev","test_alpha1","test_alpha2","test_alpha3","test_alpha2_orignal"]


```

```opr.py```: It is used to generate the same data for Paragraph Representation as used in the InfoTabS paper, same as  [infotabs](https://github.com/utahnlp/infotabs-code/blob/master/scripts/preprocess/json_to_para.py). This is used to generate data for testing out DRR step in the ablation. Important arguments are the same as bpr.py. Refer above.

```drr.py```: It is used to generate the data for the Distracting Row Removal (DRR) step.

```

important argument details in drr.py

--json_dir set as the directory path of the json files, i.e we set it to a folder named ./../data/tables/json/ in this case
--data_dir set to the directory containing the data after running bpr.py with mode set to 1 (with newline character between sentences) i.e. we set it to a folder named ./../../temp/drr/ in this case
--save_dir set to the directory where the generated drr data will be saved. Here we have set it to ./../../temp/data/drr/
--threshold this can be used to vary the value of our hyperparameter k (considers the top k rows)
--splits data split names for which the drr is to be generated. Default is set to all the splits [train, dev, test_alpha1, test_alpha2, test_alpha3, test_alpha2_orignal]
--sort set to 1 to arrange the sentences in order of obtained alignment scores from highest to lowest. Before adding KG explicit we additionally sort the important rows so that rows required for inference and their corresponding KG explicit are less likely to exceede BERT tokenisation limit on knowledge addition. Default value is set to 0(no sorted order)

```

```kg_explicit.py```: It is used to generate the data for the KG explicit step.

```


important argument details in kg_explicit.py

--json_dir set as the directory path of the json files, i.e we set it to a folder named ./../data/tables/json/ in this case
--data_dir set to the directory containing the data after running bpr.py with mode set to 1 (with newline character between sentences) i.e. we set it to a folder named ./../../temp/drr/ in this case
--KG_dir set to the directory containing the extracted KG explicit data (from sources such as Wordnet and Wikipedia) i.e. here we have set it to 
--threshold this can be used to vary the value of our hyperparameter k (considers the top k rows)
--splits data split names for which the drr is to be generated. Default is set to all the splits [train, dev, test_alpha1, test_alpha2, test_alpha3, test_alpha2_orignal]
--output_dir set to the directory where the generated data along with KG explicit will be stored i.e. ./../../temp/data/kg_explicit
--kg_threshold this can be used to vary the amount of knowledge added (hyperparameter k1). We will add knowledge for the keys of the top k1 rows.
--order default value is set to end. In this case the knowledge is added to the end of the paragraph otherwise it is added to the start.


```

To generate all the premises for main results from the json files, use ```json_to_all.sh```  which sequentially run the individual ```py``` files as discussed above

```

cd scripts
cd preprocess
mkdir -p ./../../temp
mkdir -p ./../../temp/data
bash json_to_all.sh

```

Now, generate the premises for ablation studies from the json file, use ```json_to_ablation.sh```  which sequentially run the individual ```py``` files as discussed above. 

```

cd scripts
cd preprocess
mkdir -p./../../temp
mkdir -p ./../../temp/data
bash json_to_ablation.sh

```

You would now see a ```temp/data/``` folder. ```temp/data/``` will contain sub-folders for several premise types. For example,

```

temp/data/
│ 
└── bpr                         # better paragraph representation
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit
│ 
└── drr                         # distracting row removal
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit
│ 
└── kg_explicit                 # addition of KG explicit
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit
│ 
└── opr                         # old paragraph representation
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit
│ 
└── drr_ablation                # DRR for ablation
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit
│ 
└── kg_explicit_ablation        # KG_explicit for ablation
    ├── dev.tsv                         # development datasplit
    ├── test_alpha1.tsv                 # test alpha1 datasplit
    ├── test_alpha2.tsv                 # test alpha2 datasplit
    ├── test_alpha3.tsv                 # test alpha3 datasplit
    └── train.tsv                       # training datasplit

```

### 2.2 Vectorizing, same as [infotabs](https://github.com/utahnlp/infotabs-code)
Then we need to batch the examples and vectorize them:


```

cd ../roberta
mkdir ../../temp/processed                         
bash preprocess_roberta.sh 

```

You would see a ```temp/processed/``` folder. ```temp/processed/``` will contain sub-folders for several premise types. For example,

```

temp/processed/
│
└── bpr                             # bpr
    ├── dev.pkl                         # development datasplit
    ├── test_alpha1.pkl                 # test alpha1 datasplit
    ├── test_alpha2.pkl                 # test alpha2 datasplit
    ├── test_alpha3.pkl                 # test alpha3 datasplit
    └── train.pkl                       # training datasplit

```

### 3. Training and Prediction with RoBERTa

```

mkdir -p ./../../temp/models/

```
The above directory will contain all the trained models

For training and prediction on the RoBERTa baseline look at the ```.\scripts\roberta\classifier.sh```:

example argument in ```train_classifier```

```

python3 roberta_classifier.py \
    --mode "train" \
    --epochs 15 \
    --batch_size 8 \
    --in_dir "./../../temp/processed/bpr/" \
    --embed_size 1024 \
    --model_dir "./../../temp/models/bpr/" \
    --model_name "model_6_0.7683333333333333" \
    --save_dir "./../../temp/models/" \
    --save_folder "bpr/" \
    --nooflabels 3 \
    --save_enable 0 \
    --eval_splits dev test_alpha1\
    --seed 13 \
    --parallel 0
    --kg "none"

```

important argument details which could be reset as needed for training and prediction


```

-- mode: set "train" for training, set "test" for prediction
-- epochs: set training epochs number (only used while training, i.e., model is "train")
-- batch_size: set batch size for training (only used while training)
-- in_dir: set as preprocessed directory name, i.e., a folder named in temp/processed/ . Use this for setting the appropriate premise type. (only used while training, i.e., model is "train") 
-- embed_size: set embedding size, i.e., (768/1024). Use 768 for Bert-small and 1024 for Bert-Large.
-- model_dir: use the model directory containing the train model (only used while prediction, i.e., model is "test")
-- model_name: model finename usually is in format 'model_<batch_number>_<dev_accuracy>' (only used while prediction, i.e., model is "test")
-- save_folder: name the primary models directory appropriately as ./../../temp/models/ (only used while training i.e., model is "train")
-- save_dir: name the primary models directory appropriately, usually same as the in_dir final directory (only used while training, i.e., model is "train")
-- nooflabels: set as 3 as three labels entailment, neutral and contradiction)
-- save_enable: set as 1 to save prediction files as predict_<datsetname>.json in model_dir. json contains accuracy, predicted label and gold label (in the same sequence order as the dataset set tsv in temp/data/)  (only used while prediction, i.e., model is "test")
-- eval-splits: ' '  separated datasplits names [dev, test_alpha1, test_alpha2, test_alpha3] (only used while prediction, i.e., model is "test")
-- seed: set a particular seed
-- parallel:  for a single GPU, 1 for multiple GPUs (used when training large data, use the same flag at both predictions and train time)
-- kg: for adding KG_implicit, use "implicit" else use "none"

```

The commands in the file ```roberta_classifier.sh``` can be referred to for replicating the results of ```Table 2``` and ```Table 3``` in the paper. Seeds used to replicate paper results can be found in the same bash file as well.

After training you would see a ```temp/models/``` folder. ```temp/models/``` will contain sub-folders for several premise types. Furthermore, prediction would create ```predict_<split>.json``` files. For example,

```

temp/models/
│
└── bpr                                             # bpr
    ├── model_<epoch_no>_<dev_accuracy>             # save models after every epoch
    ├── scores_<epoch_no>_dev.json                  # development prediction json results
    ├── scores_<epoch_no>_test.json                 # test alpha1 prediction json results
    └── predict_<split>.json                        # prediction json (when predicting with argument "-- save_enable" set to 1)

```

## Recommended Citations

```

@inproceedings{gupta-etal-2020-infotabs,
    title = "{INFOTABS}: Inference on Tables as Semi-structured Data",
    author = "Gupta, Vivek  and
      Mehta, Maitrey  and
      Nokhiz, Pegah  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.210",
    pages = "2309--2324",
    abstract = "In this paper, we observe that semi-structured tabulated text is ubiquitous; understanding them requires not only comprehending the meaning of text fragments, but also implicit relationships between them. We argue that such data can prove as a testing ground for understanding how we reason about information. To study this, we introduce a new dataset called INFOTABS, comprising of human-written textual hypotheses based on premises that are tables extracted from Wikipedia info-boxes. Our analysis shows that the semi-structured, multi-domain and heterogeneous nature of the premises admits complex, multi-faceted reasoning. Experiments reveal that, while human annotators agree on the relationships between a table-hypothesis pair, several standard modeling strategies are unsuccessful at the task, suggesting that reasoning about tables can pose a difficult modeling challenge.",
}

@inproceedings{neeraja-etal-2021-infotabskg,
    title = "Incorporating External Knowledge to Enhance Tabular Reasoning",
    author = "J. Neeraja  and
      Gupta, Vivek  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics",
    month = june,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}

```

