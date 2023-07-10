#### Conflict and duplicate paired datasets with the following filenames and data characterstics.
---
| **Dataset**   | **# Conflict (C)** | **# Duplicate (D)** | **# Neutral (N)** | ** Filename ** | 
|---------------|-------------------|--------------------|------------------|---------------------------|
| CDN           | 5,553             | 1,673              | 3,400            | cdn_pairs.csv|
| WorldVista    | 10,843            | -                  | 35               | world_vista_clean_pairs.csv  |
| UAV           | 6,652             | -                  | 18               | uav_clean_pairs.csv | 
| PURE          | 2,191             | -                  | 20               | pure_clean_pairs.csv       |
| OPENCOSS      | 6,776             | -                  | 10               | open_coss_clean_pairs.csv         |
| CN            | 5,553             | -                  | 3,400            | cn_pairs.csv          | 

#### For CDN dataset, implement the following code to see the structure and class distribution of datasets:
```python
# Load the dataset
df = pd.read_csv('/Data/cdn_pairs.csv')
df['Class'] = df['Class'].replace(['Conflict', 'Duplicate,'Neutral'], [0,1,2])
df.Class.value_counts()
```

#### For rest of the datasets, implement following code:
```python
# Load the dataset
df = pd.read_csv('/Data/uav_clean_pairs.csv')
df['Class'] = df['Class'].replace(['Conflict', 'Neutral'], [1, 0])
df.Class.value_counts()
```
#### NER_Data
---
This folder containes train and test files use to train a software-specific model used in `Code/Rule-based` codes. Please follow the instructions given in `Code/Rule-based/README.md` to utilise these data files.
