### cross_domain_main.py contains training code for cross-data experiments.
The following code can be used to run the file:
```python
!python  /content/cross_domain_main.py \
--test_name 'wv' \
--train_name 'pure' 'uav' \
--test_path /content/world_vista_clean_pairs.csv \
--train_path  /content/pure_clean_pairs.csv  /content/uav_clean_pairs.csv \
--combination 'double' \
--tokenizer_path 'textattack/bert-base-uncased-MNLI' \
--model_path 'textattack/bert-base-uncased-MNLI' \
--num_labels 2
```
Here make sure don't use CDN dataset as test or training data because it has 3 labels and code has been set up for different datasets with 2 labels only.
* test_name - name of the testing datasets. Choices can be ('uav' , 'pure', 'open', 'cn', 'wv')
* train_name - name of the training datasets. Here you can use up to four datasets or different combination of datasets upto 4 values from the above-listed ones separate by space.
* test_path - path of testing data
* train_path - path of training datasets separted by space
* combination - training dataset combinations. choices can be ('single', 'double', 'triple', 'quad'). choose according to whatever no of training datasets you are mentioning in train_name argument.
* tokenizer_path and model_path can be any huggingface transformer checkpoint. Please make sure you put the exact path mentioned in huggingface.
* num_label - labels in the dataset.
