import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import cuda
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }
def print_final_report(full_report, cf_matrix):
    class_label = ['0','1','macro avg','weighted avg']
    df_final = pd.DataFrame(columns=['class_label', 'Precision_mean',"Precision_std", 'Recall_mean', 'Recall_std',
                       'F1-score_mean','F1-score_std','support_mean'],index = range(0,4))
    for index, i in enumerate(class_label):
        p = []
        r = []
        f = []
        s = []
        for j in range(3):
            p.append(full_report[j][i]['precision'])
            r.append(full_report[j][i]['recall'])
            f.append(full_report[j][i]['f1-score'])
            s.append(full_report[j][i]['support'])
        row = [i,np.mean(p),np.std(p),np.mean(r),np.std(r),np.mean(f),np.std(f),np.mean(s)]
        df_final.loc[index] = row
    results_cf = np.round(np.stack([cf_matrix[0],cf_matrix[1],cf_matrix[2]]).mean(axis=0),1)
    np.set_printoptions(suppress=True)
    return df_final,results_cf


def get_model():
  model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels,ignore_mismatched_sizes=True)
  return model

def get_combinations(args):
    """Create combinations of dataset for training"""
    if args.combination == 'single':
        train_df = pd.read_csv(args.train_path)
        train_df['Class'] = train_df['Class'].replace(['Conflict','Neutral'],[1,0])
        print(train_df.Class.value_counts())
        return train_df
    elif args.combination == 'double':
        data_1,data_2 = args.train_path
        train_1 = pd.read_csv(data_1)
        train_2 = pd.read_csv(data_2)
        train_df = pd.concat([train_1,train_2])
        train_df['Class'] = train_df['Class'].replace(['Conflict','Neutral'],[1,0])
        print(train_df.Class.value_counts())
        return train_df

    elif args.combination == 'triple':
        data_1,data_2,data_3 = args.train_path
        train_1 = pd.read_csv(data_1)
        train_2 = pd.read_csv(data_2)
        train_3 = pd.read_csv(data_3)
        train_df = pd.concat([train_1,train_2,train_3])
        train_df['Class'] = train_df['Class'].replace(['Conflict','Neutral'],[1,0])
        print(train_df.Class.value_counts())
        return train_df
        # Code for triple datasets
    elif args.combination == 'quad':
        data_1,data_2,data_3,data_4 = args.train_path
        train_1 = pd.read_csv(data_1)
        train_2 = pd.read_csv(data_2)
        train_3 = pd.read_csv(data_3)
        train_4 = pd.read_csv(data_4)
        train_df = pd.concat([train_1,train_2,train_3,train_4])
        train_df['Class'] = train_df['Class'].replace(['Conflict','Neutral'],[1,0])
        print(train_df.Class.value_counts())
        return train_df

    else:
        raise ValueError("Invalid combination type. Expected 'single', 'double', or 'triple'.")


# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_name', type=str, required= True,help='Choose test dataset')
parser.add_argument('--train_name', type=str, nargs='+', required = True, help='Choose training datasets except CDN . Provide datasets separated by space.')
parser.add_argument('--test_path', type=str, required=True,help='Path to the test dataset csv file.')
parser.add_argument('--train_path', type=str, nargs='+', required=True,help='Paths to the training dataset csv files. Provide paths separated by space.')
parser.add_argument('--combination', type=str, choices=['single', 'double', 'triple','quad'], required=True,help='Choose combination type. Can be "single", "double", or "triple".')
parser.add_argument('--tokenizer_path', type=str, default='textattack/bert-base-uncased-MNLI', help='The path of the tokenizer to be used')
parser.add_argument('--model_path', type=str, default='textattack/bert-base-uncased-MNLI', help='The path of the model to be used')
parser.add_argument('--num_labels', type=int, default=2, help='data labels should be listed')


args = parser.parse_args()
train_df = get_combinations(args)
train,val = train_test_split(train_df, test_size = 0.02, stratify = train_df.Class, shuffle = True, random_state = 0)
# Read the CSV file
test_data = pd.read_csv(args.test_path)
test_data['Class'] = test_data['Class'].replace(['Conflict','Neutral'],[1,0])
print(test_data.Class.value_counts())
# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=100,ignore_mismatched_sizes=True)
model_path = args.model_path



training_args = TrainingArguments(
    output_dir='/content/output',
    do_train=True,
    do_eval=True,
    num_train_epochs=5,
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./CDN-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    fp16=True,
    save_total_limit = 1,
    metric_for_best_model='F1',
    load_best_model_at_end=True,
    run_name=args.tokenizer_path+"_CDN"
)

# training on train_df
val_encodings = tokenizer(list(val.Text1),list(val.Text2), truncation=True, padding=True)
val_dataset = MyDataset(val_encodings, list(val.Class))
train_encodings = tokenizer(list(train.Text1),list(train.Text2), truncation=True, padding=True)
train_dataset = MyDataset(train_encodings, list(train.Class))
trainer = Trainer(model_init=get_model,tokenizer=tokenizer,train_dataset = train_dataset,args=training_args,eval_dataset=val_dataset,
                    compute_metrics= compute_metrics, callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()
# testing on other dataset
test_encodings = tokenizer(list(test_data.Text1),list(test_data.Text2), truncation=True, padding=True)
test_dataset = MyDataset(test_encodings, list(test_data.Class))
pred=trainer.predict(test_dataset)
pred2=np.argmax(pred.predictions, axis=1)
cm=confusion_matrix(test_data.Class,pred2, labels= [0,1])
print("*************************************************************************************\n")
print(cm)
print("*************************************************************************************\n")
print(classification_report(test_data.Class,pred2, labels=[0,1], digits = 8))
