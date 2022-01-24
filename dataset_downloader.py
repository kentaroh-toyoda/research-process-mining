import os
import json
import urllib.request
import zipfile
import gzip
import shutil
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
        if re.match(r'.*\.zip$', dataset['filename']):
            print('It is being decompressed', dataset['name'])
            with zipfile.ZipFile(full_path) as zip_ref:
                zip_ref.extractall(dataset_dir)
        elif re.match(r'.*\.gz$', dataset['filename']):
            # make a filename for output 
            tmp = re.search(r'(.*)\.gz$', dataset['filename'])
            output_filename = dataset_dir + tmp.group(1)
            with gzip.open(full_path, 'rb') as f_in:
                with open(output_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
