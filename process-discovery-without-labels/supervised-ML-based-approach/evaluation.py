import os
import re
import glob
import pickle
import random 
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import batch_feature_extraction

def evaluate(i, d, r_train=0.8, r_other=5, n_estimators=100):
    # d: dataset containing features with 'dataset' and 'class'
    print('Iteration:', i + 1, '/', n_iter)
    # find the number of datasets
    all_dataset_names = set(d['dataset'])
    # number of datasets
    n_datasets = len(all_dataset_names)
    n_train = int(round(n_datasets * r_train))
    # extract the names of datasets
    dataset_names_train = set(random.sample(list(all_dataset_names), n_train))
    # use the rest for test
    dataset_names_test = all_dataset_names - dataset_names_train
    # training dataset
    train_DF = (d[d['dataset'].isin(dataset_names_train)])
    # the samples of 'other' are too many! let's sample some.
    # say the number of other classes' samples is N, then sample r_other*N
    n_major_classes = len(train_DF[train_DF['class'] == 'case:concept:name'])
    n_samples_other = int(round(r_other * n_major_classes))
    train_DF_main_classess = train_DF[~train_DF['class'].isin(['other'])]
    train_DF_other = train_DF[train_DF['class'].isin(['other'])]
    train_DF_other_sampled = train_DF_other.sample(n=n_samples_other, replace=False)
    train_DF = pd.concat([train_DF_main_classess, train_DF_other_sampled])
    # check if this operation is correct
    print(Counter(train_DF['class']))
    # labels
    y_train = np.array(train_DF['class'])
    # features
    X_train = np.array(train_DF.drop(['dataset', 'class'], axis=1))
    test_DF = (d[d['dataset'].isin(dataset_names_test)])
    # labels
    # test dataset
    y_test = np.array(test_DF['class'])
    # features
    X_test = np.array(test_DF.drop(['dataset', 'class'], axis=1))
    # train a classifier
    classifier = RandomForestClassifier(n_estimators=n_estimators)
    classifier.fit(X_train, y_train)
    # evaluate the classifier
    # NOTE: we don't just compare each result, rather 
    # determine a 'majority' class from and compare it with a true class.
    test_DF['predicted'] = classifier.predict(X_test)
    tmp = test_DF[['dataset', 'class', 'predicted']]
    # a majority vote
    result = tmp.groupby(['dataset', 'class']).agg(lambda x: x.value_counts().index[0]).reset_index()
    # print summary of a result
    print(classification_report(result['class'], result['predicted']))
    # return f1 score of each class
    return result

if __name__ == '__main__':
    # save timestamp
    now = datetime.now()
    current_datetime = now.strftime('%d/%m/%Y %H:%M:%S')
    # number of iteration
    n_iter = 100
    # n_rows: specify the number of rows read from each dataset for feature extraction
    n_rows_set = [10, 50, 100, 500, 1000]
    # r_train: specify how much data are used for training (default: 80%)
    r_train_set = [0.8]
    # r_other: how much samples of 'other' are used for *training* compared to the major three classes (i.e. case_id, activity, and timestamp)
    # Note that the number of samples of the three major classes is same
    r_other_set = [1, 2.5, 5, 7.5, 10]
    # n_estimators: number of trees in RF
    n_estimators_set = [10, 50, 100, 500]
    # specify datasets
    datasets = glob.glob('../../datasets/*.csv', recursive=False)
    # output_path: where results are saved
    output_path = './results.csv'
    for n_rows in n_rows_set:
        # feature extraction
        d = pd.concat([batch_feature_extraction(dataset, n_rows) for dataset in datasets])
        for r_train in r_train_set:
            for r_other in r_other_set:
                for n_estimators in n_estimators_set:
                    results = pd.concat([evaluate(i, d, r_train, r_other, n_estimators) for i in range(n_iter)])
                    # store the used parameters
                    results['n_rows'] = n_rows
                    results['r_train'] = r_train
                    results['r_other'] = r_other
                    results['RF_n_estimators'] = n_estimators
                    results['evaluation_datetime'] = current_datetime # not sure if needed
                    print(results)
                    results.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))

