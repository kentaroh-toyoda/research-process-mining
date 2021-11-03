import os
import json
import urllib.request
import zipfile
import re

dataset_dir = './datasets/'

# make directory if not exists
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# datasets are listed in datasets.json
with open('datasets.json') as f:
  datasets = json.load(f)

# download the datasets
for dataset in datasets:
    full_path = dataset_dir + dataset['filename']
    # if the dataset has been downloaded, do nothing
    if os.path.exists(full_path):
        print(dataset['name'], 'has been already downloaded.')
    else:
        print('Downloading', dataset['name'])
        # obtain a dataset
        urllib.request.urlretrieve(dataset['url'], full_path)
        # unzip it if it's a zip file
        if re.match(r'*.zip$', dataset['filename']):
            with zipfile.ZipFile(full_path) as zip_ref:
                zip_ref.extractall(dataset_dir)
