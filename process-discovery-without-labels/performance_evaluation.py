import os
import subprocess
import glob

datasets = glob.glob('../datasets/*.csv')
miners = ['inductive_miner', 'heuristics_miner']
methods = ['GA', 'bruteforce']
round = '1'
results_path = './results'
pops = ['10']
gens = ['100']
evaluation_methods = ['CV', 'no split']
metrics = ['fitness', 'precision', 'generalization' , 'simplicity', 'average', 'Buijs2014']

# batch evaluation
for dataset in datasets:
    for miner in miners:
        for method in methods:
            for evaluation_method in evaluation_methods:
                for metric in metrics:
                    print('dataset: ' + dataset + ', method: ' + method + ', miner: ' + miner)
                    subprocess.run(['python', './main.py', 
                        '--dataset', dataset,
                        '--miner', miner,
                        '--round', round,
                        '--method', method,
                        '--pop', pops[0],
                        '--gen', gens[0],
                        '--n_max_traces', '10',
                        '--n_max_activities', '50',
                        '--evaluation_method', evaluation_method,
                        '--results_path', results_path,
                        '--metric', metric
                        ])

# single run test
#  dataset = '../datasets/BPIC13_cp.csv'
#  subprocess.run(['python', './main.py', 
    #  '--dataset', dataset,
    #  '--miner', miners[0], 
    #  '--round', round,
    #  '--method', methods[1], 
    #  '--pop', pops[0],
    #  '--gen', gens[0],
    #  '--n_max_traces', '10',
    #  '--n_max_activities', '50',
    #  '--evaluation_method', evaluation_methods[0],
    #  '--results_path', results_path,
    #  '--metric', metrics[4]
    #  ])
