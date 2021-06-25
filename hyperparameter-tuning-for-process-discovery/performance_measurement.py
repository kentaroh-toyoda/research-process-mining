import os
import subprocess
import glob

datasets = glob.glob('../datasets/*.xes')
miners = ['inductive_miner', 'heuristics_miner']
#  methods = ['cboa', 'mpsboa', 'msgtboa']
methods = ['cboa']
round = '1'
results_path = './results'
n_samples = '25'

for dataset in datasets:
    for miner in miners:
        for method in methods:
            print('dataset: ' + dataset + ', method: ' + method + ', miner: ' + miner)
            subprocess.run(['python', './hyperparameters_tuning.py', 
                '--event-log-path', dataset,
                '--miner', miner, 
                '--round', round,
                '--method', method, 
                '--results-path', results_path,
                '--n-samples', n_samples])
