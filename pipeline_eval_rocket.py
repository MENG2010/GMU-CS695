"""Pipeline - train and evaluate baseline rocket.

@author: Ying
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import numpy as np
import pickle
import time
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sktime.transformations.panel.rocket import Rocket, MiniRocket

from utils.data import load_dataset

datasets = ['CBF', 'Coffee', 'DistalPhalanxTW', 'ECG5000', 'ECGFiveDays', 'FaceAll', 
            'GunPoint', 'InsectWingbeatSound', 'MiddlePhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Plane', 'ShapeletSim',
            'Trace', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']

num_rounds = 5
num_kernels = 10000
n_jobs = 6

def train(dataset):
    # load training dataset
    print(f'[INFO] Loading training dataset...')
    X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
    print(f'[INFO] type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    # extract features using Rocket kernels
    print(f'[INFO] Extracting features...')
    start_time = time.monotonic()
    feat_extractor = Rocket(num_kernels=num_kernels, n_jobs=n_jobs)
    feat_extractor.fit(X_train)
    feature_train = feat_extractor.transform(X_train)
    extract_cost = time.monotonic() - start_time
    print(f'[INFO] extract_cost: {extract_cost}')
    print(f'[INFO] feature_train.shape: {feature_train.shape}; {feature_train[0][:5]}')
    
    # fit a classifier
    print(f'[INFO] Training rocket-ridge classifier on {dataset}...')
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    )
    # rocket = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10))
    classifier.fit(feature_train, y_train)
    fit_cost = time.monotonic() - start_time - extract_cost
    train_cost = extract_cost + fit_cost
    print(f'[INFO] fit_cost: {fit_cost}; train_cost: {train_cost}')
    
    score = classifier.score(feature_train, y_train)
    print(f'[INFO] train score: {score}')
    
    return feat_extractor, classifier, train_cost, score, len(y_train)


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
    eval_scores = [
        ['model', 'dataset', 'round', 'acc_train', 'acc_test', 'time_train', 'time_test', 'train_size', 'test_size']
    ]
    eval_analysis = [
        ['model', 'dataset', 'acc_train_mean', 'acc_test_mean', 'time_train_mean', 'time_test_mean']
    ]
    for dataset in datasets:
        print(f'[INFO] Evaluating on {dataset}')
        eval_rounds = [0, 0, 0, 0]  # acc_train, acc_test, time_train, time_test
        eval_path = os.path.join('exp', 'eval', dataset)
        Path(eval_path).mkdir(parents=True, exist_ok=True)
        
        for r in range(num_rounds):
            model_path = os.path.join('exp', 'models', dataset, f'round-{r}')
            Path(model_path).mkdir(parents=True, exist_ok=True)
            # Path(os.path.join(eval_path, f'round-{r}')).mkdir(parents=True, exist_ok=True)
            
            extractor, classifier, time_train, acc_train, train_size = train(dataset=dataset)
            # save trained model
            print(f'[INFO] Save trained model to {model_path}')
            pickle.dump(extractor, open(os.path.join(model_path, 'feat-rocket.pkl'), 'wb'))
            pickle.dump(classifier, open(os.path.join(model_path, 'rocket-ridge.pkl'), 'wb'))
            
            # feat_extractor = pickle.load(open('exp/models/feat-rocket.pkl', 'rb'))
            # classifer = pickle.load(open('exp/models/rocket-ridge.pkl', 'rb'))
            time_test, acc_test, test_size = eval(extractor, classifier, dataset=dataset)
            eval_rounds[0] += acc_train
            eval_rounds[1] += acc_test
            eval_rounds[2] += time_train
            eval_rounds[3] += time_test
            
            eval_scores.append(['rocket-ridge', dataset, f'round-{r}', acc_train, acc_test, time_train, time_test, train_size, test_size])
            print(f'[INFO] scores:\n{eval_scores[0]}\n{eval_scores[-1]}')
            
        eval_analysis.append(['rocket-ridge', dataset, eval_rounds[0]/num_rounds, eval_rounds[1]/num_rounds, eval_rounds[2]/num_rounds, eval_rounds[3]/num_rounds])
        filename = f'eval_analysis-rocket-{dataset}.csv'
        filepath = os.path.join(eval_path, filename)
        print(f'[INFO] Save evaluation analysis to {filepath}')
        report = pd.DataFrame(eval_analysis[-2:], columns=eval_analysis[0])
        report.to_csv(filepath, index=False)
        print('='*80)
        
    filename = 'eval_scores-rocket.csv'
    filepath = os.path.join('exp/eval', filename)
    print(f'[INFO] Save evaluation scores to {filepath}')
    report = pd.DataFrame(eval_scores[1:], columns=eval_scores[0])
    report.to_csv(filepath, index=False)
    print('[INFO] Done evaluating Rocket-Ridge.')
    
    filename = f'eval_analysis-rocket.csv'
    filepath = os.path.join('exp/eval',, filename)
    print(f'[INFO] Save evaluation analysis to {filepath}')
    report = pd.DataFrame(eval_analysis[1:], columns=eval_analysis[0])
    report.to_csv(filepath, index=False)