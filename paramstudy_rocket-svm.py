"""Pipeline - train and evaluate Rocket+SVM.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import numpy as np
import pickle
import time
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sktime.transformations.panel.rocket import Rocket

from utils.data import load_dataset

effective_datasets = ['CBF', 'Coffee', 'DistalPhalanxTW', 'ECG5000', 'ECGFiveDays', 'FaceAll', 
            'GunPoint', 'InsectWingbeatSound', 'MiddlePhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Plane', 'ProximalPhalanxTW', 'ShapeletSim',
            'Trace', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']
ineffective_datasets = ['CinCECGTorso', 'Earthquakes', 'Ham', 'Herring', 'MedicalImages', 'Phoneme', 'ScreenType', 'WordSynonyms', 'Worms']
datasets = effective_datasets + ineffective_datasets
# datasets = ['Coffee', 'ECG5000']

num_rounds = 5
num_kernels = 10000
n_jobs = 10

out_dir = 'exp/param_study'
report_dir = 'exp/param_study/results'
model_dir = 'exp/param_study/models'

extractor_name = 'rocket'
model_name = 'svm'
report_pattern = f'{extractor_name}-{model_name}-tuning'


def tune_parameters(dataset):
    # load training dataset
    print(f'[INFO] Loading training dataset...')
    X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
    print(f'[INFO] type: {type(X_train)}; X_train.shape: {X_train.shape}')
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    # extract features using Rocket kernels
    print(f'[INFO] Extracting features...')
    start_time = time.monotonic()
    feat_extractor = Rocket(num_kernels=num_kernels, n_jobs=n_jobs)
    feat_extractor.fit(X_train)
    feature_train = feat_extractor.transform(X_train)
    extract_cost = time.monotonic() - start_time
    print(f'[INFO] extract_cost: {extract_cost}')
    print(f'[INFO] feature_train.shape: {feature_train.shape}; {feature_train[0][:5]}')
    
    param_grid = [{'svc__tol': [1e-3, 1e-5, 1e-6], 
                  'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
                  'svc__C': [0.001, 0.01, 0.1, 1]
                  }]
        
    # fit a classifier
    print(f'[INFO] Training rocket-svm classifier on {dataset}...')
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        SVC(gamma='scale')
    )
    
    estimators = GridSearchCV(classifier, param_grid, cv=5, n_jobs=n_jobs).fit(feature_train, y_train)
    
    # estimators.best_params_
    # estimators.best_score_
    # estimators.best_estimator_
    
    return estimators, feat_extractor
    

def eval(feat_extractor, classifier, dataset):
    # evaluate on test dataset
    print(f'[INFO] Evaluating model...]')
    X_test, y_test = load_dataset(name=dataset, split='test', return_X_y=True)
    start_time = time.monotonic()
    feature_test = feat_extractor.transform(X_test)
    score = classifier.score(feature_test, y_test)
    test_cost = time.monotonic() - start_time
    print(f'[INFO] test score: {score}; cost: {test_cost}')
    return test_cost, score, len(y_test)

    
if __name__ == '__main__':
    best_scores = [['dataset', 'acc_train', 'acc_test', 'best_params']]
    
    for dataset in datasets:
        print(f'[INFO] Evaluating on {dataset}')
        eval_path = os.path.join(report_dir, dataset)
        Path(eval_path).mkdir(parents=True, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'svm', dataset)
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