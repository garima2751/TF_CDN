To run `rule_based_aa_ner_main.py`, first you need to train the software-specific NER model using the following code:
```python
! python run_ner.py \
  --model_name_or_path bert-base-uncased \
  --train_file  /Data/NER_Data/AA_train_1.json\
  --validation_file  Data/NER_Data/AA_test_1.json \
  --test_file Data/NER_Data/AA_test_1.json \
  --output_dir /output/ner-bert \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir
```
The path given in **output_dir** is the path of saved NER model and please note this path for further training.
Now, to run rule_based_aa_ner_main.py, use following code:
```python
!python  /content/rule_based_aa_ner_main.py \
--data_path  /content/pure_sample.csv  \
--ner_model_output_path output_dir\
--final_output_path /content/final_results.csv
```
* *data_path*: Path of the false postives predictions data points obtained from cross-dataset experiments. Basically, you can save the predictions obtained from `Cross-Domain` codes and save it as csv file and pass it here.
* *ner_model_output_path*: Path of saved NER model shown the above command
* *final_output_path*: Provide a path to save the output from the code in csv format.

To run `rule_based_srl_pos_main.py`, use following code:
```python
!python  /content/rule_based_srl_pos_main.py \
--data_path  /Data/pure_sample.csv  \
--rule_choice 'SRL' \
--final_output_path /output/final_results.csv
```
* *data_path*: Path of the false postives predictions data points obtained from cross-dataset experiments. Basically, you can save the predictions obtained from `Cross-Domain` codes and save it as csv file and pass it here.
* *rule_choice*: SRL or POS
* *final_output_path*: Provide a path to save the output from the code in csv format.
