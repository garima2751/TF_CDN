import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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


# Initialize argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, help='The name of the dataset to be used')
parser.add_argument('--csv_path', type=str, default='/content/open_coss_clean_pairs.csv', help='The path of the csv file to be used')
parser.add_argument('--tokenizer_path', type=str, default='textattack/bert-base-uncased-MNLI', help='The path of the tokenizer to be used')
parser.add_argument('--model_path', type=str, default='textattack/bert-base-uncased-MNLI', help='The path of the model to be used')
parser.add_argument('--num_labels', type=int, default=2, help='data labels should be listed')

args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv_path)
if args.data_name in ['open','uav','wv','pure']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral'],[1,0])
    print(df.Class.value_counts())
elif args.data_name == 'cdn':
    df['Class'] = df['Class'].replace(['Conflict','Duplicate','Neural'],[0,1,2])
    print(df.Class.value_counts())
elif args.data_name == 'cn':
    df=df[df.Class!="Duplicate"]
    df['Class'] = df['Class'].replace(['Conflict','Neural'],[1,0])
    df.reset_index(drop=True, inplace=True) # check first
    print(df.Class.value_counts())
else:
    print("wrong dataset name \n")


# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=100,ignore_mismatched_sizes=True)
model_path = args.model_path

# Define the training arguments
# Your code here

# Your code for cross-validation and training here

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

device = 'cuda' if cuda.is_available() else 'cpu'
full_report = []
cf_matrix = []
skf = StratifiedKFold(n_splits=3, random_state = 1,shuffle = True)
t = df.Class
for train_index, test_index in skf.split(np.zeros(len(t)), t):
  train_data = df.loc[train_index]
  test_data = df.loc[test_index]
  print("entering fold ****************\n")
  print(train_data['Class'].value_counts(),test_data['Class'].value_counts())
  print("The shape of folds are :",train_data.shape,test_data.shape)
  train_encodings = tokenizer(list(train_data.Text1),list(train_data.Text2), truncation=True, padding=True)
  test_encodings = tokenizer(list(test_data.Text1),list(test_data.Text2), truncation=True, padding=True)
  train_dataset = MyDataset(train_encodings, list(train_data.Class))
  test_dataset = MyDataset(test_encodings, list(test_data.Class))
  trainer = Trainer(model_init=get_model,tokenizer=tokenizer,train_dataset = train_dataset,args=training_args,eval_dataset=test_dataset,
    compute_metrics= compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
  trainer.train()
  pred=trainer.predict(test_dataset)
  pred2=np.argmax(pred.predictions, axis=1)
  print("**************************************************************************************\n")
  cm=confusion_matrix(test_data.Class, pred2, labels= [0,1])
  cf_matrix.append(cm)
  print(cm)
  print("**************************************************************************************\n")
  full_report.append(classification_report(test_data.Class, pred2, labels=[0,1],output_dict=True, digits = 8))
  print(classification_report(test_data.Class, pred2, labels=[0,1],digits = 8))
  print("**************************************************************************************\n")

#***************************************************************************************************
#print final report
final_df, final_cf_matrix = print_final_report(full_report,cf_matrix)
print(final_df)
final_df.to_csv('/content/output/final_output.csv')
