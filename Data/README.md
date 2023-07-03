This directory includes all the conflict and duplicate paired datasets with the following filenames and data characterstics.
| **Dataset**   | **# Conflict (C)** | **# Duplicate (D)** | **# Neutral (N)** | ** Filename ** | 
|---------------|-------------------|--------------------|------------------|---------------------------|
| CDN           | 5,553             | 1,673              | 3,400            | ConfDubNo2.csv|
| WorldVista    | 10,843            | -                  | 35               | world_vista_clean_pairs.csv  |
| UAV           | 6,652             | -                  | 18               | uav_clean_pairs.csv | 
| PURE          | 2,191             | -                  | 20               | pure_clean_pairs.csv       |
| OPENCOSS      | 6,776             | -                  | 10               | open_coss_clean_pairs.csv         |
| CN            | 5,553             | -                  | 3,400            | -           | 

For CN dataset, implement following code to convert CDN dataset to CN dataset:
```python
# Load the dataset
df = pd.read_csv('/Data/ConfDubNo2.csv')
df = df[df.Class != "Duplicate"]
df.reset_index(drop=True, inplace=True)
df['Class'] = df['Class'].replace(['Conflict', 'Neutral'], [1, 0])
df.Class.value_counts()

For rest of the datasets, implement following code:
```python
# Load the dataset
df = pd.read_csv('/Data/uav_clean_pairs.csv')
df['Class'] = df['Class'].replace(['Conflict', 'Neutral'], [1, 0])
df.Class.value_counts()
