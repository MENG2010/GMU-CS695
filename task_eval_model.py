"""Evaluate baseline models.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import argparse
import os

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.kernel_based import MiniRocketClassifier
from sklearn.metrics import accuracy_score

from utils.file import parse_yaml
from utils.data import load_dataset


def get_model(model, dataset):
    pass


def eval(model_configs, data, exp_configs):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baseline models.')
    parser.add_argument('--config', type=str, required=False,
                        default='configs/eval_baseline.yml',
                        help='path to evaluating baseline models config file.')
    parser.add_argument('--datasets', type=str, required=False,
                        default='configs/datasets.yml', 
                        help='path to datasets to evaluate.')
    
    args = parser.parse_args()
    configs = parse_yaml(args.config)
    datasets = parse_yaml(args.datasets)
    
    baselines = configs['exp']['models']
    for model_name in baselines:
        model_path = os.path.join(configs['exp']['model_base'],
                                  model_name,
                                  configs[model_name]['filename'])
        for dataset in datasets['datasets']:
            print(f'[INFO] evaluating {model_name} model on {dataset}.')
            eval(
                model_configs=configs[model_name],
                data=data,
                exp_configs=configs['exp']
            )