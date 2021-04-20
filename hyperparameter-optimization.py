import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('logfile', type=str, help='the file name of an event log')

if __name__ == "__main__":
    args = parser.parse_args()
    # import xes file
    event_log = pm4py.read_xes(args.logfile)
    # apply a process discovery method
    process_tree = inductive_miner.apply(event_log)
    # evaluate performance metrics

event_log = pm4py.read_xes('./datasets/BPIC12.xes')
process_tree = inductive_miner.apply(event_log)
subprocess.call(['java', ''])
