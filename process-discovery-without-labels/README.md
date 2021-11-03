# Overview

A series of codes for process discovery without labels

# Usage

- `main.py`: the main body of the proposal that uses optimizer. It also contains a bruteforce method where all candidates are evaluated. I'll probably rename this file.
- `performance_evaluation.py`: a controller for `main.py`. It runs single- or batch-execution with parameters specified.

The following python files are supplementary

- `characteristics_of_labels.py`: preliminary test code to see what kind of characteristics can be seen from event logs
- `log_converter.py`: convert XES, which is a standard data format for event logs, files into CSV.
