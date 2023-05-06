"""Pipeline - parameter study of minirocket-cnn model.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import pickle
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sktime.transformations.panel.rocket import MiniRocket
from sktime.classification.deep_learning.cnn import CNNClassifier

from utils.data import load_dataset, reshape

effective_datasets = ['DistalPhalanxTW', 'FaceAll', 'InsectWingbeatSound', 'MiddlePhalanxOutlineCorrect', 'OliveOil', 'ProximalPhalanxTW', 'ShapeletSim', 'UWaveGestureLibraryY']
ineffective_datasets = ['Earthquakes', 'Ham', 'MedicalImages', 'Phoneme', 'ScreenType', 'Worms']
# datasets = effective_datasets + ineffective_datasets
datasets = ['DistalPhalanxTW', 'Earthquakes']

out_base = 'eval/param_study'
num_rounds = 5
num_kernels = 10000
n_jobs = 6
n_epochs = 1000


def get_extractor(X_train):
    extractor = MiniRocket(num_kernels=num_kernels, n_jobs=n_jobs)
    extractor.fit(X_train)
    
    return extractor
    

def train(extractor, trainset):
    # load training dataset
    # print(f'[INFO] Loading training dataset...')
    # X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
    
    X_train, y_train = trainset
    start_time = time.monotonic()
    # fit a classifier
    X_train = extractor.transform(X_train).to_numpy()
    X_train = reshape(X_train, extractor='minirocket')
    print(f'[INFO] X_train.type: {type(X_train)}; X_train.shape: {X_train.shape}')
    
    classifier = CNNClassifier(
        kernel_size=2,
        n_conv_layers=1,
        n_epochs=n_epochs, 
        activation='linear',
        verbose=0)
        
    classifier.fit(X_train, y_train)
    train_cost = time.monotonic() - start_time
    score = classifier.score(X_train, y_train)
    print(f'[INFO] train_cost: {train_cost}; train score: {score}')
    
    return classifier, train_cost, score, len(y_train)


def eval(extractor, classifier, dataset):
    # evaluate on test dataset
    print(f'[INFO] Evaluating model...]')
    X_test, y_test = load_dataset(name=dataset, split='test', return_X_y=True)
    start_time = time.monotonic()
    X_test = extractor.transform(X_test).to_numpy()
    X_test = reshape(X_test, extractor='minirocket')
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
    for dataset in datasets:
        print(f'[INFO] Evaluating on {dataset}')
        eval_rounds = [0, 0, 0, 0]  # acc_train, acc_test, time_train, time_test
        eval_path = os.path.join(out_base, 'eval', dataset)
        Path(eval_path).mkdir(parents=True, exist_ok=True)
        
        for r in range(num_rounds):
            model_path = os.path.join(out_base, 'models', 'minirocket-cnn', dataset, f'round-{r}')
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            X_train, y_train = load_dataset(name=dataset, split='train', return_X_y=True)
            extractor_filepath = os.path.join(model_path, 'feat-minirocket.pkl')
            if os.path.exists(extractor_filepath):
                print(f'[INFO] Load trained extractor from {extractor_filepath}]')
                extractor = pickle.load(open(extractor_filepath, 'rb'))
            else:
                print(f'[INFO] Train extractor...]')
                extractor = get_extractor(X_train)
                pickle.dump(extractor, open(extractor_filepath, 'wb'))
            
            print(f'[INFO] Train model on {dataset}...')
            classifier, time_train, acc_train, train_size = train(extractor=extractor, trainset=(X_train, y_train))
            # save trained model
            print(f'[INFO] Save trained model to {model_path}')
            # pickle.dump(extractor, open(os.path.join(model_path, 'feat-minirocket.pkl'), 'wb'))
            pickle.dump(classifier, open(os.path.join(model_path, 'minirocket-cnn.pkl'), 'wb'))
            
            time_test, acc_test, test_size = eval(extractor, classifier, dataset=dataset)
            eval_rounds[0] += acc_train
            eval_rounds[1] += acc_test
            eval_rounds[2] += time_train
            eval_rounds[3] += time_test
            
            eval_scores.append(['minirocket-cnn', dataset, f'round-{r}', acc_train, acc_test, time_train, time_test, train_size, test_size])
            print(f'[INFO] scores:\n{eval_scores[0]}\n{eval_scores[-1]}')
            
        eval_analysis.append(['minirocket-cnn', dataset, eval_rounds[0]/num_rounds, eval_rounds[1]/num_rounds, eval_rounds[2]/num_rounds, eval_rounds[3]/num_rounds])
        filename = f'eval_analysis-minirocket-cnn-{dataset}.csv'
        filepath = os.path.join(eval_path, filename)
        print(f'[INFO] Save evaluation analysis to {filepath}')
        report = pd.DataFrame(eval_analysis[-1:], columns=eval_analysis[0])
        report.to_csv(filepath, index=False)
        print('='*80)
        
    filename = 'eval_scores-minirocket-cnn.csv'
    filepath = os.path.join(out_base, filename)
    print(f'[INFO] Save evaluation scores to {filepath}')
    report = pd.DataFrame(eval_scores[1:], columns=eval_scores[0])
    report.to_csv(filepath, index=False)
    print('[INFO] Done evaluating rocket-cnn.')
    
    filename = f'eval_analysis-minirocket-cnn.csv'
    filepath = os.path.join(out_base, filename)
    print(f'[INFO] Save evaluation analysis to {filepath}')
    report = pd.DataFrame(eval_analysis[1:], columns=eval_analysis[0])
    report.to_csv(filepath, index=False)
