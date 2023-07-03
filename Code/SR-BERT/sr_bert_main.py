import tensorflow as tf
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers import losses
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import classification_report,  confusion_matrix

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


parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, help='The name of the dataset to be used')
parser.add_argument('--csv_path', type=str, default='/content/open_coss_clean_pairs.csv', help='The path of the csv file to be used')
parser.add_argument('--sbert_model_path', type=str, default='all-mpnet-base-v2', help='The path of the model to be used')
parser.add_argument('--num_labels', type=int, default=2, help='data labels should be listed')
parser.add_argument('--output_path', type=str, default='/content/final_output.csv', help='output path for final results')

args = parser.parse_args()
#print("arguments taken \n")

df = pd.read_csv(args.csv_path)
#print("df extracted ")
if args.data_name in ['open','uav','wv','pure']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral'],[1,0])
    class_labels = [0,1]
    print(df.Class.value_counts())
elif args.data_name == 'cdn':
    df['Class'] = df['Class'].replace(['Conflict','Duplicate','Neural'],[0,1,2])
    class_labels = [0,1,2]
    print(df.Class.value_counts())
elif args.data_name == 'cn':
    df=df[df.Class!="Duplicate"]
    df['Class'] = df['Class'].replace(['Conflict','Neural'],[1,0])
    class_labels = [0,1]
    df.reset_index(drop=True, inplace=True) # check first
    print(df.Class.value_counts())
else:
    print("wrong dataset name \n")


model=SentenceTransformer(args.sbert_model_path)
loss = losses.SoftmaxLoss(model=model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=args.num_labels)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps=10000, decay_rate=0.95, staircase=True)


full_report = []
cf_matrix = []
skf = StratifiedKFold(n_splits=3, random_state = 1,shuffle = True)
t = df.Class
for train_index, test_index in skf.split(np.zeros(len(t)), t):
  train_data = df.loc[train_index]
  test_data = df.loc[test_index]
  print("**************************************************************************************\n")
  print("Entering into folds \n")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]

  train_examples= [InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)  for e in train_data.iterrows()]
  train_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)  for e in train_data.iterrows()]

  test_examples=[InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)    for e in test_data.iterrows()]
  test_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)   for e in test_data.iterrows()]


  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
  test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=16)


  train_evaluator = LabelAccuracyEvaluator(train_dataloader, softmax_model=loss)
  test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=loss)
  model.fit(train_objectives=[(train_dataloader, loss)], epochs=1, warmup_steps=100)
  print("**************************************************************************************\n")
  print("After transformer checkpoint test performance \n")
  print(model.evaluate(test_evaluator))
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = [tf.keras.callbacks.EarlyStopping(patience=3),tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)]

  print("*****************************************************************************************\n")
  print("Keras model preparing ......\n")

  test1_enc=model.encode(test_data.Text1.values)
  test2_enc=model.encode(test_data.Text2.values)
  test_diff= np.subtract(test1_enc, test2_enc)
  testIN= np.concatenate((test1_enc,test2_enc, test_diff), axis=1)
  tr1_enc=model.encode(train_data.Text1.values)
  tr2_enc=model.encode(train_data.Text2.values)
  tr_diff= np.subtract(tr1_enc, tr2_enc)
  trainIN= np.concatenate((tr1_enc,tr2_enc, tr_diff), axis=1)
  y_train = keras.utils.to_categorical(train_data.Class ,num_classes = args.num_labels)
  y_test = keras.utils.to_categorical(test_data.Class,num_classes = args.num_labels)
  in_emb = keras.Input(shape=(2304,), name="in")
  x = layers.Dense(1500, activation="relu")(in_emb)
  x = layers.Dropout(0.2)(x)
  output = layers.Dense(args.num_labels, activation="softmax")(x)
  cls_model = keras.Model(in_emb, output)
  opt = keras.optimizers.Adam(learning_rate=lr_schedule)
  cls_model.compile(optimizer=opt, loss= tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
  cls_model.fit(trainIN ,y_train,  epochs=50, validation_data=(testIN, y_test), callbacks=[model_checkpoint_callback])

  pred=np.argmax(cls_model.predict(testIN), axis=1)
  real= test_data.Class.values
  print("*****************************************************************************************\n")
  print(confusion_matrix(real, pred))
  cf_matrix.append(confusion_matrix(real, pred))
  print("*****************************************************************************************\n")
  print(classification_report(real, pred, labels=class_labels, digits = 6))
  full_report.append(classification_report(real, pred, labels=class_labels,output_dict = True, digits = 6))

final_df, final_cf_matrix = print_final_report(full_report,cf_matrix)
print(final_df,final_cf_matrix)
final_df.to_csv(args.output_path)
