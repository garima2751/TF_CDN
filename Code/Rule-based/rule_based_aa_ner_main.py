from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import argparse
import pandas as pd
#from utils_rule_based import performance_report
def get_actor_action(tags):
  actor_action = []
  for i,d in enumerate(tags):
    if(d['entity'] in ['B-Actor','I-Actor','B-Action','I-Action']):
      actor_action.append((d['word'],d['entity']))
  return actor_action

def get_similar_action(t_aa_1,t_aa_2):
  similar_action = []
  for token_1,tag_1 in t_aa_1:
    if 'B-Action' in tag_1 or 'I-Action' in tag_1:
      for token_2,tag_2 in t_aa_2:
        if token_1 == token_2:
          similar_action.append((token_1,tag_1))
  return similar_action

def find_similar_actor(t_aa_1,t_aa_2):
  similar_actor = []
  for token_1,tag_1 in t_aa_1:
    if 'B-Actor' in tag_1 or 'I-Actor' in tag_1:
      for token_2,tag_2 in t_aa_2:
        if token_1 == token_2:
          similar_actor.append((token_1,tag_1))
  return similar_actor

def get_prediction_aa_ner(original_df):
    df = original_df.copy()
    df['second_label'] = 0
    for index, row in df.iterrows():
      tags_1 = ner_model(row['Text1'])
      t_aa_1 = get_actor_action(tags_1)
      tags_2 = ner_model(row['Text2'])
      t_aa_2 = get_actor_action(tags_2)
      sim_actor = find_similar_actor(t_aa_1,t_aa_2)
      sim_action = get_similar_action(t_aa_1,t_aa_2)
      if len(sim_actor) == 0:
        if len(sim_action) == 0:
          df.loc[index,'second_label'] = 1
      elif len(sim_actor) != 0:
        if len(sim_action) == 0:
          df.loc[index,'second_label'] = 1
    print("The reclassification results using Actor-Action:",df.second_label.value_counts())
    return df

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, help='The path of the data for reclassification')
parser.add_argument('--ner_model_output_path', type=str, help='Software-specific trained ner model path ')
parser.add_argument('--final_output_path',type= str,help='Path where the final data after reclassification')

args = parser.parse_args()

model = AutoModelForTokenClassification.from_pretrained(args.ner_model_output_path)
tokenizer = AutoTokenizer.from_pretrained(args.ner_model_output_path)

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

df = pd.read_csv(args.data_path)

re_class_df = get_prediction_aa_ner(df)

re_class_df.to_csv(args.final_output_path)
