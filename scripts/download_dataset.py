# if you want to download the dataset, you can run this script:
# '''python download_dataset.py'''

# if you meet with some network problems, you can set the mirror site before running the script:
# export HF_ENDPOINT=https://hf-mirror.com

import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset name', default= "thuml/UTSD")
parser.add_argument('--subset', type=str, help='subset name', default= "UTSD-1G")
parser.add_argument('--output', type=str, help='output directory', default='data/pretrain/utsd')

args = parser.parse_args()
if not args.output:
    raise SystemExit('--output is required')

ds = datasets.load_dataset(args.dataset, args.subset)

# the dataset have not been divided into train, test, and val splits
# therefore, ds['train'] contains all the time series
# you can split them by yourself, or use our default split as train:val=9:1 in '''utsdataset.py'''
all = ds['train']

# print the total number of time series
print(f'total {len(all)} single-variate series')

# each item is a single-variate series containing: 
# 1. dataset name (item_id)
# 2. start time (start)
# 3. end time (end)
# 4. sampling frequecy (freq)
# 5. time series values (target)
# timestampes are optional since some datasets are irregular and may not have 

# see https://huggingface.co/datasets/thuml/UTSD/viewer for more details`
print(all[0].keys())

# you can access the time series values by item['target']
num_timepoints = len(all[0]['target'])
print(f'the first time series containing {num_timepoints} time points')

# or generate the timestamps by item['start'], item['end'], and item['freq']
all.save_to_disk(args.output)
print(f'save the dataset to {args.output}')

