### sr_bert_main.py contains training code for SR_BERT model.

The following code provides the details on how to train the SR_BERT model:
```python
! python  /content/sr_bert_main.py  --data_name 'cdn'
                                    --csv_path  /content/ConfDubNo2.csv
                                    --sbert_model_path 'distilbert-base-nli-mean-tokens'
                                    --num_labels 3
                                    --output_path /content/final_results.csv
```
* Here, the file paths is according to Google colab working, if you are running the code in your own local directory , you need to specify the local directory path where the respectives files are present.
* data_name choices ('uav', 'open' , 'wv', 'cdn', 'cn', 'pure')
* csv_path - file path for the dataset (Dataset can be takne from data folder in this repo)
* sbert_model_path - sbert pre-trained model checkpoint name (Please refer the following link to choose the checkpoints [SBERT pre-trained checkpoint](https://www.sbert.net/docs/pretrained_models.html))
* num_labels should be 3 for CDN and rest all datasets have 2 labels

### For Paper Results:
* To replicate the results in the paper from SR-BERT paper, please train SBERT checkpoint (`distilbert-base-nli-mean-tokens') for 5 epochs and perform 5 fold cross validation.
* Changes to epoch value and fold value can be easily done on sr_bert_main.py file
* Also, the sample code output and running commands has been provided in the `example_notebooks' folder.

