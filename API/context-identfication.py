import os
import re
import glob
import time
import random 
import pickle
import signal
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
from random import sample
from datetime import datetime
from itertools import product
from collections import Counter
from SetSimilaritySearch import all_pairs
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# for CV
# process mining
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
# miners
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
# performance metrics
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

def get_candidates(features, target_class, k=2):
    # read the trained classifier for the target_class
    clf = pickle.load(open('../models/' + target_class + '.clf', 'rb'))
    # transform the features in DF to np.array
    X = np.array(np.transpose(features), dtype=object)
    # only return the highest k candidates
    probs = clf.predict_proba(X)
    # the probs contain two probabilities (for binary classficiation), but we need to check
    # which element refers to the target class's probability
    target_class_index = np.where(clf.classes_ == target_class)
    # print('target_class_index', target_class_index, 'of', clf.classes_)
    probs = [probs[i][target_class_index] for i in range(len(probs))]
    # print(probs)
    top_probs = sorted(probs)[-k:]
    # a candidate must be at least identified as a target key (i.e. judge if a prob > 0.5)
    top_probs = [prob for prob in top_probs if prob > 0.5]
    cands = np.where([p in top_probs for p in probs])[0].tolist()
    # print(cands)
    return cands

def measure_score(d=None, key_indices=None, miner='inductive_miner', metric='generalization', n_splits=2,
                  model_discovery_timeout_in_sec=5, evaluation_timeout_in_sec=60):
    def discover_model(event_log, miner):
        # default parameters are used
        if miner == 'inductive_miner':
            net, im, fm = inductive_miner.apply(event_log,
                {pm4py.algo.discovery.inductive.variants.im.algorithm.Parameters.NOISE_THRESHOLD: 0.2},
                pm4py.algo.discovery.inductive.algorithm.Variants.IM)
        elif miner == 'heuristics_miner':
            net, im, fm = heuristics_miner.apply(event_log, {
                heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
                heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65,
                heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 1,
                heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 1,
                heuristics_miner.Variants.CLASSIC.value.Parameters.DFG_PRE_CLEANING_NOISE_THRESH: 0.05,
                heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH: 2})
        return net, im, fm

    def evaluate_score(event_log, net, im, fm, metric):
        if metric == 'fitness':
            score = replay_fitness_evaluator.apply(event_log, net, im, fm, 
                    variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['log_fitness']
        elif metric == 'precision':
            score = precision_evaluator.apply(event_log, net, im, fm, 
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        elif metric == 'generalization':
            score = generalization_evaluator.apply(event_log, net, im, fm)
        elif metric == 'simplicity':
            score = simplicity_evaluator.apply(net)
        else:
            if metric == 'Buijs2014':
            # A paper "Quality dimensions in process discovery: The importance of fitness, 
            # precision, generalization and simplicity" proposed to calculate the following 
            # four metrics with giving 10 times more weight to replay fitness than 
            # the other three.
            # 10 x + 3x = 1 => x = 1 / 13
                weights = [10/13, 1/13, 1/13, 1/13]
            elif metric == 'average':
                weights = [0.25, 0.25, 0.25, 0.25]
            fitness = replay_fitness_evaluator.apply(event_log, net, im, fm, 
                    variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['log_fitness']
            precision = precision_evaluator.apply(event_log, net, im, fm, 
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
            #  fitness = replay_fitness_evaluator.apply(event_log, net, im, fm, 
                    #  variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)['log_fitness']
            #  precision = precision_evaluator.apply(event_log, net, im, fm, 
                    #  variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            generalization = generalization_evaluator.apply(event_log, net, im, fm)
            simplicity = simplicity_evaluator.apply(net)
            score = np.dot(weights, [fitness, precision, generalization, simplicity])
        return score
    
    def timeout_handler(signum, frame):
        raise Exception('timeout')
        
    def cross_validation(event_log_df, train, test):
        # dataset need be sorted by timestamp (https://pm4py.fit.fraunhofer.de/documentation#item-import-csv)
        #  print("%s %s" % (train, test))
        train_log_df = event_log_df.iloc[train]
        try:
            train_log_df = train_log_df.sort_values('time:timestamp')
        except:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'the identified timestamp seems wrong', 0)
            return 0
            
        test_log_df = event_log_df.iloc[test]
        try:
            test_log_df = test_log_df.sort_values('time:timestamp')
        except:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'the identified timestamp seems wrong', 0)
            return 0
        
        train_log = log_converter.apply(train_log_df, \
                                        variant=log_converter.Variants.TO_EVENT_LOG)
        test_log = log_converter.apply(test_log_df, \
                                       variant=log_converter.Variants.TO_EVENT_LOG)
    
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), \
              'discovering a process model...')
        
        # discover_model may take time. hence, we set a timeout for it
        # https://stackoverflow.com/a/494273/7184459
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(model_discovery_timeout_in_sec)
        try:
            net, im, fm = discover_model(train_log, miner)
        except Exception:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'gave up discovering a model')
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'score evaluated:', 0)
            return 0
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'process model discovered')
        # reset a timer when a model discovery finishes before the deadline
        signal.alarm(0)
        # set another timer for evaluation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(evaluation_timeout_in_sec)
        try:
            score = evaluate_score(test_log, net, im, fm, metric)
        except Exception:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'gave up evaluating a model')
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'score evaluated:', 0)
            return 0
        signal.alarm(0)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'score evaluated:', score)
        return score
    # assume the columns as case_id, timestamp, and activity from left
    event_log_df = d.iloc[:, key_indices]
    event_log_df.columns = ['case:concept:name', 'time:timestamp', 'concept:name']
    # evaluate the goodness with CV
    gkf = GroupKFold(n_splits=n_splits)
    X = list(event_log_df.index)
    groups = list(event_log_df['case:concept:name'])
    if len(groups) < n_splits:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'the case_id seems wrong')
        return 0
    scores = [cross_validation(event_log_df, train, test) \
          for train, test in gkf.split(X, groups=groups)]
    score = round(np.mean(scores), ndigits=3)
    print('final score:', score)
    return score

# feature_extraction: 
# input: values in a column
# output: features
def feature_extraction(values, round_digits=3):
    def local_feature_extraction(value):
        patterns = ['[a-z]', '[A-Z]', '\d', '\s', '[^a-zA-Z\d\s]']
        f = {}
        if len(value) == 0:
            f['f_chars'] = 0 
            f['f_words'] = 0
            for p in patterns:
                f['f_{}'.format(p)] = 0
        else:
            # length: length of a value
            f['f_chars'] = len(value)
            # words: number of words in a value
            f['f_words'] = len(re.findall(r'\w+', value))
            # The following code find the frequency of each pattern in patterns in a value
            for p in patterns:
                f['f_{}'.format(p)] = round(len(re.findall(p, value)) / len(value), round_digits)
        return f

    # set type of values string
    values = values.astype(str)
    # local features
    f_local = [local_feature_extraction(value) for value in values]
    # convert it into a DF to easily calculate mean of each feature
    f = pd.DataFrame.from_dict(f_local)
    f = f.apply(np.mean, axis=0)
    # global features
    # count the occurence of each value in values
    counts = Counter(values).values()
    if len(counts) > 1:
        # f_ratio_unique_values: how much unique values are involved
        f['f_ratio_unique_values'] = round(len(set(values)) / len(values), round_digits)
        # f_mean_unique_values: mean value of number of appearance of each value
        f['f_mean_unique_values'] = round(statistics.mean(counts), round_digits)
    else:
        f['f_ratio_unique_values'] = 1
        f['f_mean_unique_values'] = 1
    return f.to_numpy()

def identify_attributes(event_log, k=2, max_trials=5, miner='inductive_miner', metric='generalization', in_detail=True):
    # extract features
    features = event_log.apply(lambda x: feature_extraction(x), axis=0)
    found_cands = False
    ####### Stage 1 #######
    for i_trial in range(max_trials):
        case_id_cand = get_candidates(features, 'case:concept:name', k=k)
        timestamp_cand = get_candidates(features, 'time:timestamp', k=k)
        activity_cand = get_candidates(features, 'concept:name', k=k)
        print('case_id_cand:', case_id_cand)
        print('timestamp_cand:', timestamp_cand)
        print('activity_cand:', activity_cand)
        if all([len(timestamp_cand) > 0, len(case_id_cand) > 0, len(activity_cand) > 0]):
            found_cands = True
            break
    if not found_cands:
        print('Could not find key column candidates. Halt.')
        return [None, None, None]

    # if we can identify a single candidate for each, simply return them
    if all([len(case_id_cand) == 1, len(timestamp_cand) == 1, len(activity_cand) == 1]):
        return [case_id_cand[0], timestamp_cand[0], activity_cand[0]]
    # otherwise, we proceed to the second stage
    ####### Stage 2 #######
    else:
        # list the possible combinations
        combinations = [list(t) for t in product(case_id_cand, timestamp_cand, activity_cand) if len(set(t)) == 3]
        # print(combinations)
        # scoring
        scores = [measure_score(event_log, c, miner=miner, metric=metric, n_splits=2) for c in combinations]
        if in_detail:
            return scores, combinations
        else:
            # only return the combination with the highest score
            return [combinations[i] for i in np.argwhere(np.amax(scores) == scores).flatten().tolist()][0]