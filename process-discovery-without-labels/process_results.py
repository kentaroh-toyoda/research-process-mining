import glob
import pandas as pd
import numpy as np

# read all results
result_files = glob.glob('./results/*.pickle')
# each pickle is stored in an array and is a dict. need to convert them to df
tmp = [pd.DataFrame.from_dict(pd.read_pickle(r)[0]) for r in result_files]
# concat them into one df
results = pd.concat(tmp)
# add a column to identify an attribute 
results['attribute'] = np.tile(np.array(['case_id', 'activity', 'timestamp']), len(result_files))
results

# stats
# add a column that indicates if an identified label is correct
results = results.assign(is_correct=lambda x: x.labels == x['identified labels'])
# print all results
pd.set_option('display.max_rows', None, 'display.max_columns', None)
# grouping for summary
grouped = results.groupby(['dataset', 'miner', 'method', 'evaluation_method', 'attribute'])
# accuracy
grouped['is_correct'].agg([('accuracy', lambda x: sum(x) / len(x))])
# time
grouped = results.groupby(['dataset', 'miner', 'method', 'evaluation_method'])
grouped['elapsed_time_in_sec'].agg([('computation_time', 'mean')])
