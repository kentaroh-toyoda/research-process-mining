import re
import glob
import random
import pickle
import statistics
import numpy as np
import pandas as pd
from collections import Counter
from collections import OrderedDict

# feature_extraction: 
# input: expect a vector of values in a column
# output: set of features
def local_feature_extraction(value):
    patterns = ['[a-z]', '[A-Z]', '\d', '\s', '[^a-zA-Z\d\s]']
    if len(value) == 0:
        f = OrderedDict([
            ('length', 0), 
            ('words', 0)
            ])
        for p in patterns:
            f['f_{}'.format(p)] = 0
    else:
        # length: length of value
        # words: return words separate by a space (can be captured by \w+
        f = OrderedDict([
            ('length', len(value)), 
            ('words', len(re.findall(r'\w+', value))),
            ])
        # The following code find the frequency of each pattern in patterns in a value
        for p in patterns:
            f['f_{}'.format(p)] = round(len(re.findall(p, value)) / len(value), 3)
    return f

def feature_extraction(values):
    # set type of values string
    values = values.astype(str)
    # local features
    f_local = [local_feature_extraction(value) for value in values]
    # convert it into DF
    f = pd.DataFrame.from_dict(f_local)
    # global features
    # count the occurence of each value in values
    counts = Counter(values).values()
    if len(counts) > 1:
        # find the mean and variance of counts
        # append global features
        f['ratio_unique_values'] = round(len(set(values)) / len(values), 3)
        f['mean_unique_values'] = round(statistics.mean(counts), 3)
        f['var_unique_values'] = round(statistics.variance(counts), 3)
    else:
        f['ratio_unique_values'] = 1
        f['mean_unique_values'] = 1
        f['var_unique_values'] = 0
    return f

nrows = 1000
def batch_feature_extraction(dataset):
    print('Processing', dataset)
    # read a csv log file
    event_log = pd.read_csv(dataset, nrows=nrows)
    # replace NaN with '' so that an evaluation function can work
    event_log = event_log.fillna(np.nan).replace([np.nan], [''])
    # h: header
    h = event_log.columns.values.tolist()
    # check if a dataset contains case_ids, activities, and timestamps
    if (('case:concept:name' in h) and ('concept:name' in h) and ('time:timestamp' in h)):
        labels = [
            h.index('case:concept:name'),
            h.index('concept:name'),
            h.index('time:timestamp')
        ]
        n_columns = event_log.columns.size
        # pick one column that is not case_id, activity, or timestamp 
        # for 'other' class in supervised ML (if the number of columns is more than three)
        if n_columns > 3:
            other_indices = set(range(n_columns)) - \
                    set([
                        h.index('case:concept:name'), 
                        h.index('concept:name'), 
                        h.index('time:timestamp')
                        ])
            # feature extraction
            f_caseid = feature_extraction(event_log['case:concept:name'])
            f_caseid['class'] = 'case:concept:name'
            f_activity = feature_extraction(event_log['concept:name'])
            f_activity['class'] = 'concept:name'
            f_timestamp = feature_extraction(event_log['time:timestamp'])
            f_timestamp['class'] = 'time:timestamp'
            # extract featues of each 'other' column
            f_other = [feature_extraction(event_log[h[index]]) for index in other_indices]
            f_other = pd.concat(f_other)
            f_other['class'] = 'other'
            # concatenate features and labels
            f = pd.concat([f_caseid, f_activity, f_timestamp, f_other])
            # add a dataset name
            f['dataset'] = dataset
            return f

if __name__ == '__main__':
    datasets = glob.glob('../../datasets/*.csv', recursive=False)
    tmp = [batch_feature_extraction(dataset) for dataset in datasets]
    # merge features + classes of all datasets
    f = pd.concat(tmp)
    print(f.head)
    f.to_pickle('./dataset.pkl')
