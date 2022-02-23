# Learning Bill Similarity with Annotated and Augmented Corpora of Bills
[**Paper**](https://aclanthology.org/2021.emnlp-main.787/) | [**Download**](#Download) | [**Install**](#Install) | [**Data**](#Data)  | [**Models**](#Models) | [**Train**](#Train) | [**Predict**](#Predict) | [**Result**](#Result) |
[**Citation**](#Citation) | [**Contact**](#Contact)


This repository contains the source code for the paper [Learning Bill Similarity with Annotated and Augmented Corpora of Bills](https://aclanthology.org/2021.emnlp-main.787/), which is accepted by EMNLP 2021. Please refer to the paper for more details.

---

## 🔽Download
Please download 1) datasets and 2) our best model first before running our code.

### How to get Datasets
1) download datasets from [**this link**](https://drive.google.com/drive/folders/1-WbZ5Lw3OTyMRLT3lLCRK680cG0DJk00?usp=sharing)
2) unzip `synthetic_pairs.zip` and `subsection_list.zip`
3) locate them in `data/subsection_pairs/synthetic_pairs/train_sub_sec_pairs.csv` and `data/subsection_list/subsection_list.[csv/pkl]` respectively

<br>

### How to apply our Best Model
1) download our best model from [**this link**](https://drive.google.com/drive/folders/1-WbZ5Lw3OTyMRLT3lLCRK680cG0DJk00?usp=sharing)
2) unzip `trained_model.zip`
3) locate the model in `trained_model/roberta-base_best_subsect_sim_model`

---

## 🛠Install
We recommend ***Python 3.6*** for training our model.
```bash
pip install -e .
pip install -r requirements.txt
```
---
## 🗃Data

### Human-annotated & Synthetic datasets
We use both 1) human-labeled (`data/subsection_pairs/human_annotated_pairs`) and 2) synthetic datasets (`data/subsection_pairs/synthetic_pairs`) for training. The human-annotated data instances are split into train, validation, and test set with the ratio of 7:1:2. We use synthetic data for training only. The size of synthetic pairs differs depending on experiment settings.


|   Dataset Type  |     Train     | Validation | Test |
|:---------------:|:-------------:|:----------:|:----:|
| human-annotated |         3,305 |        472 |  944 |
|    synthetic    | more than 10K |      -     |   -  |

<br>

### Synthetic Pair Generation
The code of synthetic subesction pair generation is in `data/synthetic_subsection_pair_generation/generate_synthetic_subsection_pair.py`.
For generating ablation datasets, please refer annotation in function `apply_synonym_swap()` and `apply_synonym_swap_for_class3()`.

<br>

### Subsection text
The preprocessed subsection texts are in `data/subsection_list/subsection_list.[csv/pkl]`(csv/pkl version).

<br>

### common N-grams
The list of common N-grams in subsections is in `data/common_ngrams_subsection`.

---
## 🤖Models 
### Pretrined models
We use three pre-trained models provided by [*Hugging Face*](https://huggingface.co/).
1) roberta-base
2) bert-base-cased 
3) nlpaueb/legal-bert-base-uncased

<br>

### Our Best Model
Our best model is stored in `trained_model/roberta-base_best_subsect_sim_model`

<br>

### The code of our Model
Our code is based on cross encoder in [*Sentence Transformer*](https://github.com/UKPLab/sentence-transformers).
The code of our customized model is in `sentence_transformers/cross_encoder/CrossEncoder_parallel.py`

---
## 🏃Train
There are three types of training.
1) train a model with human-annotated subsection pairs
2) train a model with synthetic subsection pairs
3) retrain the model from 2 with human-annotated pairs (second-stage training)

<br>

### 1) Train with Human-annotated Dataset
* Argument Description

Multi-GPUs can be used for training. We use 4 GPUs as below (`[0,1,2,3]`). 

Results reported in our paper are the average of five trials. You can set  the nth trial using `nth_result`. 

For each run, you can leave note using `memo` argument.

* Save a  Model

When you train a model, the model is saved in `trained_model/{pretrained_model}_{train_data_name}/{train_data_size}/{lr}_{memo}_{nth_result}/epoch` for epoch.

Validation and Test results are also saved in the folder named `{pretrained_model}_{train_data_name}_{train_data_size}_{lr}_{memo}_{nth_result}_[test\val]_results.csv`.

<br>

The sample code is in `train/train_human_annotated.sh`
```bash
python -u train.py \
    --train_data_name human_annotated \
    --train_data_size 3305 \
    --pretrained_model roberta-base \
    --lr 2e-5 \
    --batch_size 32 \
    --num_epochs 4 \
    --gpu_list [0,1,2 3] \
    --nth_result 0 \
    --memo demo 
```

### 2) Train with Synthetic Dataset
The sample script is in `train/train_synthetic.sh`
```bash
python -u train.py \
    --train_data_name synthetic \
    --train_data_size 10000 \
    --pretrained_model roberta-base \
    --lr 2e-5 \
    --batch_size 32 \
    --num_epochs 4 \
    --gpu_list [0,1,2 3] \
    --nth_result 0 \
    --memo demo 
```

### 3) Second-stage Training
The sample script is in `train/train_second_stage.sh`.
We retrain the 3rd epoch of roberta-base trained from 2. (An epoch starts from 0 in our code. Results of 3rd epoch are in `epoch2` folder) 

The results are saved in `trained_model/{pretrained_model's_full_name}_human_annotated/`
```bash
python -u train.py \
    --train_data_name human_annotated \
    --train_data_size 3305 \
    --pretrained_model roberta-base_synthetic/10000/2e-05_demo_0/epoch2 \
    --lr 2e-5 \
    --batch_size 32 \
    --num_epochs 4 \
    --gpu_list [0,1,2 3] \
    --nth_result 0  \
    --memo demo 
```
---
## 🎯Predict
The sample code for subsection pair prediction using our best model is in `predict/predict_subsec_pair_label.py`. You can predict classes with your own model as well. Please see details in the code.
```bash
python predict/predict_subsec_pair_label.py
```

---

## 📊Result
### Subsection Pair Classification with Different Datasets
Multi-stage training with both synthetic and human-annotated data (Synthetic + Human) significantly improves performance for all classes.
|        Class        | Synthetic | Human | Synthetic + Human |
|:-------------------:|:---------:|:-----:|:-----------------:|
| 4 Identical         |      92.2 |  95.6 |          **96.9** |
| 3 Almost Identical  |      63.4 |  74.7 |          **77.6** |
| 2 Related           |      62.6 |  72.6 |          **76.3** |
| 1 Partially Related |      17.7 |  45.5 |          **51.9** |
| 0 Unrelated         |      84.2 |  95.8 |          **97.1** |
 Average Accuracy <br> Average Macro F1  |  73.5 <br> 64.0|  86.9 <br> 76.8|**88.9** <br> **79.9**|


---
## 📑Citation
```bibtex
@inproceedings{kim-etal-2021-learning,
    title = "Learning Bill Similarity with Annotated and Augmented Corpora of Bills",
    author = "Kim, Jiseon  and
      Griggs, Elden  and
      Kim, In Song  and
      Oh, Alice",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.787",
    doi = "10.18653/v1/2021.emnlp-main.787",
    pages = "10048--10064",
    abstract = "Bill writing is a critical element of representative democracy. However, it is often overlooked that most legislative bills are derived, or even directly copied, from other bills. Despite the significance of bill-to-bill linkages for understanding the legislative process, existing approaches fail to address semantic similarities across bills, let alone reordering or paraphrasing which are prevalent in legal document writing. In this paper, we overcome these limitations by proposing a 5-class classification task that closely reflects the nature of the bill generation process. In doing so, we construct a human-labeled dataset of 4,721 bill-to-bill relationships at the subsection-level and release this annotated dataset to the research community. To augment the dataset, we generate synthetic data with varying degrees of similarity, mimicking the complex bill writing process. We use BERT variants and apply multi-stage training, sequentially fine-tuning our models with synthetic and human-labeled datasets. We find that the predictive performance significantly improves when training with both human-labeled and synthetic data. Finally, we apply our trained model to infer section- and bill-level similarities. Our analysis shows that the proposed methodology successfully captures the similarities across legal documents at various levels of aggregation.",
}
```

## 📨Contact 
Please contact jiseon_kim@kaist.ac.kr or raise an issue in this repository.
