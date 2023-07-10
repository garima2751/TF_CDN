from allennlp.predictors.predictor import Predictor
import nltk
import pandas as pd
import numpy as np
import multiprocessing as mp
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk import pos_tag
import re
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse
def pos_tags(text):
  s_text = text.split()
  tokens_tag = pos_tag(s_text)
  return tokens_tag

def get_similar_noun(t_aa_1,t_aa_2):
  similar_action = []
  for token_1,tag_1 in t_aa_1:
    if 'N' in tag_1:
      for token_2,tag_2 in t_aa_2:
        if token_1 == token_2:
          similar_action.append((token_1,tag_1))
  return similar_action

def get_similar_verb(t_aa_1,t_aa_2):
  similar_action = []
  for token_1,tag_1 in t_aa_1:
    if 'V' in tag_1:
      for token_2,tag_2 in t_aa_2:
        if token_1 == token_2:
          similar_action.append((token_1,tag_1))
  return similar_action

def get_sre(text):
  a1 = predictor_srl.predict(sentence=text)
  return a1

def get_verb_pos(req):
  # remove stopwords also
  verbs = []
  stop_words = set(stopwords.words('english'))
  req = remove_punctuation(req)
  text = req.split()
  tokens_tag = pos_tag(text)
  #print(tokens_tag)
  for token,tag in tokens_tag:
    if 'VB' in tag:
      verbs.append(token)
    elif 'RBR' in tag:
      verbs.append(token)
  #print(verbs)
  output = [w for w in verbs if not w in stop_words]
  return output

def remove_punctuation(my_str):
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

  no_punct = ""
  for char in my_str:
    if char not in punctuations:
      no_punct = no_punct + char
  return no_punct

def get_srl_dict(verb_list,predictor_output):
  srl_dict = {}
  for key,value in predictor_output.items():
    for i in value:
      if type(i) is dict:
        for j,k in i.items():
          if j == 'verb':
            if k in verb_list:
              result = re.findall(r"\[.*?\]",str(i['description']))
              srl_dict[k] = result
  return srl_dict

def get_arguments(srl_dict):
  arg_0 = []
  arg_1 = []
  for key, value in srl_dict.items():
    for k,v in enumerate(value):
      if 'ARG0' in v:
        arg_0.append(value[k][6:-1])
      elif 'ARG1' in v:
        arg_1.append(value[k][6:-1])
  return arg_0, arg_1

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def get_prediction_pos(original_df):
    df = original_df.copy()
    df['second_label'] = 0
    for index, row in df.iterrows():
      tags_1 = pos_tags(row['Text1'])
      tags_2 = pos_tags(row['Text2'])
      sim_actor = get_similar_noun(tags_1,tags_2)
      sim_action = get_similar_verb(tags_1,tags_2)
      if len(sim_actor) == 0:
        if len(sim_action) == 0:
          df.loc[index,'second_label'] = 1
      elif len(sim_actor) != 0:
        if len(sim_action) == 0:
          df.loc[index,'second_label'] = 1
    print("The reclassification results using POS:",df.second_label.value_counts())
    return df



def process_row(row):
    sre_1 = get_sre(row['Text1'])
    sre_2 = get_sre(row['Text2'])
    vb_1 = get_verb_pos(row['Text1'])
    vb_2 = get_verb_pos(row['Text2'])
    common_verb = list(set(vb_1).intersection(vb_2))

    if len(common_verb) == 0:
        row['second_label'] = 1
    else:
        srl_dict_1 = get_srl_dict(common_verb,sre_1)
        srl_dict_2 = get_srl_dict(common_verb,sre_2)
        arg_0_a,arg_1_a = get_arguments(srl_dict_1)
        arg_0_b,arg_1_b = get_arguments(srl_dict_2)

        if len(arg_0_a) == 0 and len(arg_0_b) == 0 and len(arg_1_a) == 0 and len(arg_1_b) == 0:
            row['second_label'] = 1
        elif common_member(arg_1_a,arg_1_b) and common_member(arg_0_a,arg_0_b):
            row['second_label'] = 0
        else:
            row['second_label'] = 1

    return row

def get_prediction_srl_v2(original_df):
    df = original_df.copy()
    df['second_label'] = 0

    num_cores = mp.cpu_count()

    with mp.Pool(num_cores) as pool:
        df = pd.concat(pool.map(process_row, np.array_split(df, num_cores)))

    print("The reclassification results using SRL:",df.second_label.value_counts())
    return df

def get_prediction_srl(original_df):
    df = original_df.copy()
    df['second_label'] = 0
    for index, row in df.iterrows():
      sre_1 = get_sre(row['Text1'])
      sre_2 = get_sre(row['Text2'])
      vb_1 = get_verb_pos(row['Text1'])
      vb_2 = get_verb_pos(row['Text2'])
      common_verb = list(set(vb_1).intersection(vb_2))
      if len(common_verb) == 0:
        df.loc['second_label'] = 1
      else:
          srl_dict_1 = get_srl_dict(common_verb,sre_1)
          srl_dict_2 = get_srl_dict(common_verb,sre_2)
          arg_0_a,arg_1_a = get_arguments(srl_dict_1)
          arg_0_b,arg_1_b = get_arguments(srl_dict_2)
          if len(arg_0_a) == 0 and len(arg_0_b) == 0 and len(arg_1_a) == 0 and len(arg_1_b) == 0:
            df.loc[index,'second_label'] = 1
          elif common_member(arg_1_a,arg_1_b) and common_member(arg_0_a,arg_0_b):
            df.loc[index,'second_label'] = 0
          else:
            df.loc[index,'second_label'] = 1
    print("The reclassification results using SRL:",df.second_label.value_counts())
    return df


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, help='The path of the data for reclassification')
parser.add_argument('--rule_choice',type=str, help = 'Choose from SRL or POS for rule-based classification')
parser.add_argument('--final_output_path',type= str,help='Path where the final data after reclassification')

args = parser.parse_args()

df = pd.read_csv(args.data_path)
if args.rule_choice == 'POS':
    re_class_df = get_prediction_pos(df)
    re_class_df.to_csv(args.final_output_path)
elif args.rule_choice == 'SRL':
    predictor_srl = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    re_class_df = get_prediction_srl(df)
    re_class_df.to_csv(args.final_output_path)
