Sequential_main.py file contains code to train and fine-tune the transformer models over requirement pair datasets.
This file can be run using following code:
```python
! python  /content/sequential_main.py --data_name 'pure'
                                      --csv_path  /content/pure_clean_pairs.csv
                                      --tokenizer_path  'bert-base-uncased'
                                      --model_path 'bert-base-uncased'
                                      --num_labels 2
```

* data_name choices ['pure','open','wv','cdn','cn','uav']
* csv_path - path to the dataset.
* tokenizer_path and model_path - MNLI dataset trained checkpoints (roberta-large-mnli, deberta-base-mnli, bert-base-uncased-MNLI) 
Note: It is recommended that tokenizer_path and model_path should be taken from huggingface library.
* num_labels should be 3 in case of CDN dataset otherwise its 2 for other datasets

To run this code and sample output structure is provided in `example_notebooks' folder. 
