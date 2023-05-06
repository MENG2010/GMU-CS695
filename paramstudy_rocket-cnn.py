"""Pipeline - train and evaluate CNN.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import pickle
import time
import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sktime.transformations.panel.rocket import Rocket
from sktime.classification.deep_learning.cnn import CNNClassifier

from utils.data import load_dataset, reshape

effective_datasets = ['CBF', 'Coffee', 'DistalPhalanxTW', 'ECG5000', 'ECGFiveDays', 'FaceAll', 
            'GunPoint', 'InsectWingbeatSound', 'MiddlePhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Plane', 'ProximalPhalanxTW', 'ShapeletSim',
            'Trace', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']
ineffective_datasets = ['CinCECGTorso', 'Earthquakes', 'Ham', 'Herring', 'MedicalImages', 'Phoneme', 'ScreenType', 'WordSynonyms', 'Worms']
datasets = effective_datasets + ineffective_datasets
#datasets = ['Coffee', 'DistalPhalanxTW', 'Earthquakes']

num_rounds = 5
num_kernels = 1000
n_jobs = 10
n_epochs = 1000

out_dir = 'exp/param_study'
report_dir = 'exp/param_study/results'
model_dir = 'exp/param_study/models'

extractor_name = 'rocket'
model_name = 'cnn'
report_pattern = f'{extractor_name}-{model_name}-tuning'


def tune_parameters(dataset):
    # load training dataset
    print(f'[INFO] Loading training dataset...')
    X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
    #print(f'[INFO] type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    # fit a classifier
    print(f'[INFO] Training cnn classifier on {dataset}...')
    feat_extractor = Rocket(num_kernels=num_kernels, n_jobs=n_jobs)
    #print(f'[INFO] X_train.type: {type(X_train)}; X_train.shape: {X_train.shape}')
    feat_extractor.fit(X_train)
    X_train = feat_extractor.transform(X_train).to_numpy()
    X_train = reshape(X_train, extractor='rocket')
    print(f'[INFO] X_train.type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    param_grid = {'kernel_size': [2, 3, 5],
                  'n_conv_layers': [1, 2, 3],
                  'activation': ['linear']}
    
    classifier = CNNClassifier()
    
    estimators = GridSearchCV(classifier, param_grid, cv=5, n_jobs=n_jobs).fit(X_train, y_train)
    
    return estimators, feat_extractor


def eval(extractor, classifier, dataset):
    # evaluate on test dataset
    print(f'[INFO] Evaluating model...]')
    X_test, y_test = load_dataset(name=dataset, split='test', return_X_y=True)
    start_time = time.monotonic()
    X_test = extractor.transform(X_test).to_numpy()
    X_test = reshape(X_test, extractor='rocket')
    score = classifier.score(X_test, y_test)
    test_cost = time.monotonic() - start_time
    print(f'[INFO] test score: {score}; cost: {test_cost}')
    return test_cost, score, len(y_test)

    
if __name__ == '__main__':
    best_scores = [['dataset', 'acc_train', 'acc_test', 'best_params']]
    
    for dataset in datasets:
        print(f'[INFO] Evaluating on {dataset}')
        eval_path = os.path.join(report_dir, dataset)
        Path(eval_path).mkdir(parents=True, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'cnn', dataset)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        estimators, feat_extractor = tune_parameters(dataset=dataset)
        best_classifier = estimators.best_estimator_
        
        # save trained model
        print(f'[INFO] Save trained model to {model_path}')
        pickle.dump(feat_extractor, 
                    open(os.path.join(model_path, f'feat-{extractor_name}.pkl'), 'wb'))
        pickle.dump(best_classifier, 
                    open(os.path.join(model_path, f'{extractor_name}-{model_name}-best.pkl'), 'wb'))
        
        time_test, acc_test, test_size = eval(feat_extractor, best_classifier, dataset=dataset)
        
        best_scores.append(
            [dataset, estimators.best_score_, acc_test, estimators.best_params_]
        )
        print(f'[INFO] best scores: {estimators.best_score_}@\n{estimators.best_params_}')
        
        filename = f'track-{report_pattern}-{dataset}.csv'
        filepath = os.path.join(eval_path, filename)
        print(f'[INFO] Save evaluation analysis to {filepath}')
        report = pd.DataFrame(estimators.cv_results_)
        report.to_csv(filepath, index=False)
        print('-'*80)
    
    filename = f'best-{extractor_name}-{model_name}.csv'
    filepath = os.path.join(report_dir, filename)
    print(f'[INFO] Save evaluation scores to {filepath}')
    report = pd.DataFrame(best_scores[1:], columns=best_scores[0])
    report.to_csv(filepath, index=False)
    print(f'[INFO] Done evaluating {extractor_name}-{model_name}.')
    print('='*80)
    print()
