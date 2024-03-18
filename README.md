#### TF_CDN: Tranfer learning for conflict, duplicate, and neutral requirement pairs
[![DOI](https://zenodo.org/badge/637262324.svg)](https://zenodo.org/doi/10.5281/zenodo.10830022)
---
* This is a repository for the paper **''Transfer learning for conflict and duplicate detection in software requirement pairs''**
* A draft of the paper can be found on Arxiv: [Paper link](https://arxiv.org/pdf/2301.03709.pdf).

#### Repository Organisation
---
* Code: Folder includes four different components contains all the relevant code.
* Data: Folder includes all the requirement pair datasets used in paper experiments.
* Example: Folder contains jupyter notebook for demo purpose.
  
#### Usage with Google Colab
---
You can upload the notebook at `Examples/example_notebook.ipynb` and follow the instructions there to get started with training and evaluation of the proposed techniques. Make sure to replicate the paper results please follow the indepth description of each component in this `README.md` file. 
#### Getting Started
---
Clone this Github repository and open your command line to navigate the repository and code.
```
$ cd TF_CDN
$ TF_CDN/ cd Code
$ TF_CDN/Code/  .....
```
To download the all the necessary libraries please run the following commands
```
$pip install -r requirements.txt
```
**Note that these codes by deafult will run on CPU and it's recommended that you should use GPU services for faster execution times. Please follow the colab examples to get started.**
#### Running the Code
---
You can type 
```python Sequential/sequential_main.py --help``` to understand the arguments structure for the code files. You can do that for the files listed below to understand the type and name of each argument required to run the files.
The `Code` folder is organised into four main components as follows:
* `Code/Sequential/sequential_main.py`: code to train and fine-tune transformer model for requirement pair datasets. To run the file please follow the command indicated in `Code/Sequential/README.md` location.

* `Code/SR-BERT/sr_bert_main.py`: training code for SR_BERT model. To run the code in the file, please follow the instructions given in `Code/SR-BERT/README.md`.

* `Code/Cross-Domain/cross_domain_main.py`: Train models for cross-data experiments. Please follow the command indicated in `Code/Cross-Domain/README.md`.
  
* `Code/Rule-based/rule_based_aa_ner_main.py`: Execute the software-specific actor and action extraction from requirements for reclassification of false postives obtained in cross dataset experiments. To run the code, follow the instructions given in `Code/Rule-based/README.md`.
  
* `Code/Rule-base/rule_based_srl_pos_main.py`: Execute semantic role labeling (SRL) and part-of-speech (POS) tagging for cross dataset experiments. To run the code, follow the instructions given in `Code/Rule-based/README.md`.

#### Paper Results
---
To obtain the results reported in the paper, please train the models for longer durations and follow the individual instructions provided in each code component to replicate the paper results.
