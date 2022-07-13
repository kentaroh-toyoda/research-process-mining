import re
import glob
import random
import numpy as np
import pandas as pd
from SetSimilaritySearch import all_pairs

for dataset in glob.glob('../datasets/*.csv'):
    log_df = pd.read_csv(dataset)
    #  case_ids = log_df['case:concept:name'].head()
    #  activities = log_df['concept:name'].head()
    #  timestamps = log_df['time:timestamp'].head()
    print('dataset:', dataset)
    print(log_df[['case:concept:name', 'time:timestamp', 'concept:name']].head())
    print(log_df['case:concept:name'].apply(set).tolist())
    #  print('case_ids:', case_ids)
    #  print('activities:', activities)
    #  print('timestamps:', timestamps)
    print('if a digit cotains in case_id:', all([bool(re.search(r'\d', value)) for value in set(log_df['case:concept:name'])]))
    case_ids = set(log_df['case:concept:name'])
    # determine number of case_ids to be kept 
    n_samples = min(len(case_ids), 100)
    # determine which case_ids will be kept
    sampled_indices = random.sample(case_ids, n_samples)
    # filter out the others
    log_df = log_df[log_df['case:concept:name'].isin(sampled_indices)]
    sets = list(log_df.groupby('case:concept:name')['concept:name'].apply(list).apply(set))
    pairs = list(all_pairs(sets, similarity_func_name="cosine", similarity_threshold=0.1))
    print(np.mean([list(pairs[i])[2] for i in range(len(pairs))]))

