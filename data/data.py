# Read dataset file
import pandas as pd

dataset_to_file = {
    'gsm_8k': {
        'train': 'dataset/gsm_8k/train.jsonl',
        'test': 'dataset/gsm_8k/test.jsonl'
    },
}

def read_json(dataset, d_set):
    if dataset in dataset_to_file:
        if d_set in dataset_to_file[dataset]:
            data = pd.read_json(dataset_to_file[dataset][d_set], lines=True)
            return data['question'].tolist()
        else:
            FileNotFoundError
    else:
        FileNotFoundError