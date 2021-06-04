# A set of Transfer Bayesian optimization algorithms with non-expensive constraint handling ability #
# The implementations assume function maximization by default. For minimization, please multiply objective by -1 #
# The implementations assume that all decision variables are box-constrained in [0,1] #
import scipy as sp
import numpy as np
#  from numpy.linalg import inv
import pandas as pd
import os
import sys
import time
import re
import random
from datetime import datetime 
import pickle

from tbo import cboa
from tbo import mpsboa
from tbo import msgtboa

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
# miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
# performance metrics
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

#  from plotnine import *
#  from mizani.breaks import date_breaks
#  from mizani.formatters import date_format

import argparse
parser = argparse.ArgumentParser(description="Tune the hyperparameters of process discovery methods with Abhishek's package.")
parser.add_argument('--miner', dest='miner', type=str, help='a process discovery miner to be used')
parser.add_argument('--method', dest='method', type=str, help='a method to be tested')
parser.add_argument('--round', dest='round', type=int, help='the number of iterations')
args = parser.parse_args()

#  read a dataset
event_log = pm4py.read_xes('../datasets/BPIC12.xes')

n_sample = 500

################# Testing code #########################################
# Get input data and convert to input data #change file path
miner =args.miner
method = args.method

if miner == 'inductive_miner':
    dim = 1
    n_samples = 50
elif miner == 'heuristics_miner':
    dim = 5
    n_samples = 50

# scale_range: scale and shift a value given by Abhishek's library to actual parameter range
def scale_range(x, param_range, is_integer=True):
    if is_integer:
        return round(x * (max(param_range) - min(param_range)) + min(param_range))
    else:
        return x * (max(param_range) - min(param_range)) + min(param_range)

# f(x): input into boa, mspboa, ...
def f(x):
    if miner == 'inductive_miner':
        net, im, fm = inductive_miner.apply(event_log, 
            {pm4py.algo.discovery.inductive.variants.im.algorithm.Parameters.NOISE_THRESHOLD: x[0]},
            pm4py.algo.discovery.inductive.algorithm.Variants.IMf)
    #  elif miner == 'heuristics_miner':

    fitness = replay_fitness_evaluator.apply(event_log, net, im, fm, 
            variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['log_fitness']
    precision = precision_evaluator.apply(event_log, net, im, fm, 
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    generalization = generalization_evaluator.apply(event_log, net, im, fm)
    simplicity = simplicity_evaluator.apply(net)
    score = np.mean([fitness, precision, generalization, simplicity])
    return score

# only mspboa and msgtboa require cboa results
if not method == 'cboa':
    cboa_results_dir = './results/'
    r = re.compile('.*_cboa_output.pickle')
    cboa_outputs = list(filter(r.match, os.listdir(cboa_results_dir)))

for i in range(args.round):
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(current_datetime, '\n--- Round {} ---'.format(i+1))
    # measure computation time
    start_time = time.time()
    if method == 'cboa':
        result = cboa(f, dim, n_samples, init_sample_size=5, f_con=None)
    elif method == 'mpsboa':
        # randomly pick a cboa result
        chosen_result = random.sample(cboa_outputs, 1)[0]
        _, _, xS, yS = pickle.load(open(cboa_results_dir + chosen_result, 'rb'))
        result = mpsboa(f, dim, n_samples, xS, yS, init_sample_size=5, f_con=None)
    elif method == 'msgtboa':
        # randomly pick two cboa results
        chosen_results = random.sample(cboa_outputs, 2)
        _, _, xS1, yS1 = pickle.load(open(cboa_results_dir + chosen_results[0], 'rb'))
        _, _, xS2, yS2 = pickle.load(open(cboa_results_dir + chosen_results[1], 'rb'))
        xS = [xS1, xS2]
        yS = [yS1, yS2]
        result = msgtboa(f, dim, n_samples, xS, yS, init_sample_size=5, f_con=None)
    elapsed_time = round(time.time() - start_time)
    print('Result:', result)
    print('Elapsed time:', elapsed_time)
    # save a result
    prefix = './results/' + current_datetime + '_' + miner + '_' + method
    pickle.dump(elapsed_time, file=open(prefix + '_elapsed_time.pickle', 'wb'))
    pickle.dump(result, open(prefix + '_output.pickle', 'wb'))
