"""Task script to download datasets.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import argparse
import os

from file import parse_yaml
from data import load_dataset


def download_datasets(configs):
    """Download datasets."""
    downloaded = []
    failed = []
    dataset_base = configs['dataset_base']
    for name in configs['datasets']:
        try:
            print(f'[INFO] downloading {name} dataset to {dataset_base}')
            load_dataset(name=name, 
                         split=configs['split'],
                         return_X_y=configs['return_X_y'],
                         return_type=configs['return_type'],
                         downloaded_path=dataset_base)
            downloaded.append(name)
        except Exception as err:
            print(f'[Error] Failed to download {name} dataset.')
            failed.append(name)
    print(f'[INFO] finished downloading datasets.')
    print(f'[INFO] downloaded: {downloaded}')
    print(f'[INFO] failed: {failed}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets.')
    parser.add_argument('--config', type=str, required=False,
                        default='configs/datasets.yml', 
                        help='path to config file')
    
    args = parser.parse_args()
    configs = parse_yaml(args.config)
    
    download_datasets(configs=configs)