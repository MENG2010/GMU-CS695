"""Pipeline - train and evaluate CNN.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import pickle
import time
import pandas as pd
import ast
import warnings
warnings.filterwarnings('ignore')

from sktime.transformations.panel.rocket import Rocket
from sktime.classification.deep_learning.cnn import CNNClassifier

from utils.data import load_dataset, reshape, df_to_dict

effective_datasets = ['CBF', 'Coffee', 'DistalPhalanxTW', 'ECG5000', 'ECGFiveDays', 'FaceAll', 
            'GunPoint', 'InsectWingbeatSound', 'MiddlePhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Plane', 'ProximalPhalanxTW', 'ShapeletSim',
            'Trace', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']
ineffective_datasets = ['CinCECGTorso', 'Earthquakes', 'Ham', 'Herring', 'MedicalImages', 'Phoneme', 'ScreenType', 'WordSynonyms', 'Worms']
datasets = effective_datasets + ineffective_datasets
#datasets = ['Coffee', 'Phoneme']

num_rounds = 5
num_kernels = 10000
n_jobs = 10
n_epochs = 1000

out_dir = 'exp/optimal'
model_dir = 'exp/optimal/models'
report_dir = 'exp/optimal/results'

extractor_name = 'rocket'
model_name = 'cnn'
report_pattern = f'{extractor_name}-{model_name}-tuning'

optimal_filepath = os.path.join('exp/param_study/results', 'best-rocket-cnn.csv')

def train(dataset, configs):
    # load training dataset
    print(f'[INFO] Loading training dataset...')
    X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
    #print(f'[INFO] type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    start_time = time.monotonic()
    # fit a classifier
    print(f'[INFO] Training cnn classifier on {dataset}...')
    extractor = Rocket(num_kernels=num_kernels, n_jobs=n_jobs)
    
    extractor.fit(X_train)
    X_train = extractor.transform(X_train).to_numpy()
    X_train = reshape(X_train, extractor=extractor_name)
    print(f'[INFO] X_train.type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    classifier = CNNClassifier(
        kernel_size=configs['kernel_size'],
        n_conv_layers=configs['n_conv_layers'],
        n_epochs=n_epochs, 
        activation=configs['activation'],
        verbose=0)
        
    classifier.fit(X_train, y_train)
    train_cost = time.monotonic() - start_time
    score = classifier.score(X_train, y_train)
    print(f'[INFO] train_cost: {train_cost}; train score: {score}')
    
    return extractor, classifier, train_cost, score, len(y_train)


def eval(extractor, classifier, dataset):
    # evaluate on test dataset
    print(f'[INFO] Evaluating model...]')
    X_test, y_test = load_dataset(name=dataset, split='test', return_X_y=True)
    start_time = time.monotonic()
    X_test = extractor.transform(X_test).to_numpy()
    X_test = reshape(X_test, extractor=extractor_name)
    score = classifier.score(X_test, y_test)
    test_cost = time.monotonic() - start_time
    print(f'[INFO] test score: {score}; cost: {test_cost}')
    return test_cost, score, len(y_test)

    
if __name__ == '__main__':
    eval_scores = [
        ['model', 'dataset', 'round', 'acc_train', 'acc_test', 'time_train', 'time_test', 'train_size', 'test_size']
    ]
    eval_analysis = [
        ['model', 'dataset', 'acc_train_mean', 'acc_test_mean', 'time_train_mean', 'time_test_mean']
    ]
    
    optimal_params = pd.read_csv(optimal_filepath)[['dataset', 'best_params']].to_dict()
    optimal_params = df_to_dict(optimal_params)
    print(optimal_params)
    
    for dataset in datasets:
        print(f'[INFO] Evaluating on {dataset}')
        eval_rounds = [0, 0, 0, 0]  # acc_train, acc_test, time_train, time_test
        eval_path = os.path.join(report_dir, dataset)
        Path(eval_path).mkdir(parents=True, exist_ok=True)
        configs = ast.literal_eval(optimal_params[dataset])
        print(f'[INFO] optimal params: {configs}')
        
        for r in range(num_rounds):
            model_path = os.path.join(model_dir, dataset, f'round-{r}')
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            extractor, classifier, time_train, acc_train, train_size = train(dataset=dataset, configs=configs)
            # save trained model
            print(f'[INFO] Save trained model to {model_path}')
            pickle.dump(extractor, open(os.path.join(model_path, f'feat-{extractor_name}.pkl'), 'wb'))
            pickle.dump(classifier, open(os.path.join(model_path, f'{extractor_name}-{model_name}.pkl'), 'wb'))
            
            time_test, acc_test, test_size = eval(extractor, classifier, dataset=dataset)
            eval_rounds[0] += acc_train
            eval_rounds[1] += acc_test
            eval_rounds[2] += time_train
            eval_rounds[3] += time_test
            
            eval_scores.append([f'{extractor_name}-{model_name}', dataset, f'round-{r}', acc_train, acc_test, time_train, time_test, train_size, test_size])
            print(f'[INFO] scores:\n{eval_scores[0]}\n{eval_scores[-1]}')
            
        eval_analysis.append([f'{extractor_name}-{model_name}', 
                              dataset, 
                              eval_rounds[0]/num_rounds, 
                              eval_rounds[1]/num_rounds, 
                              eval_rounds[2]/num_rounds, 
                              eval_rounds[3]/num_rounds])
        filename = f'eval_analysis-{extractor_name}-{model_name}-{dataset}.csv'
        filepath = os.path.join(eval_path, filename)
        print(f'[INFO] Save evaluation analysis to {filepath}')
        report = pd.DataFrame(eval_analysis[-1:], columns=eval_analysis[0])
        report.to_csv(filepath, index=False)
        print('='*80)
        
    filename = f'eval_scores-{extractor_name}-{model_name}.csv'
    filepath = os.path.join(report_dir, filename)
    print(f'[INFO] Save evaluation scores to {filepath}')
    report = pd.DataFrame(eval_scores[1:], columns=eval_scores[0])
    report.to_csv(filepath, index=False)
    print(f'[INFO] Done evaluating {extractor_name}-{model_name}.')
    
    filename = f'eval_analysis-{extractor_name}-{model_name}.csv'
    filepath = os.path.join(report_dir, filename)
    print(f'[INFO] Save evaluation analysis to {filepath}')
    report = pd.DataFrame(eval_analysis[1:], columns=eval_analysis[0])
    report.to_csv(filepath, index=False)
