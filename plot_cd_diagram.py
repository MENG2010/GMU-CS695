"""Plot CD diagram.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import pandas as pd

from plot.cd_diagram import draw_cd_diagram


def param_study_plots():
    report_path = 'exp/param_study/analysis'
    filenames = [
        'summary-minirocket-svm-kernel.csv',
        'summary-rocket-svm-kernel.csv',
        'summary-minirocket-cnn-KernelSize.csv',
        # 'summary-rocket-cnn-KernelSize.csv',
        'summary-minirocket-cnn-NLayers.csv',
        'summary-rocket-cnn-NLayers.csv',
    ]

    categories = [
        'Kernel', 'Kernel', 
        'Kernel-Size', #'Kernel-Size',
        'Num-Conv-Layers', 'Num-Conv-Layers',
    ]
    sort_by = 'Dataset'
    metric = 'Mean-Val-Accuracy'

    titles = [
        'Mean-Val-Accuracy (MiniRocket-SVM, Kernel)',
        'Mean-Val-Accuracy (Rocket-SVM, Kernel)',
        'Mean-Val-Accuracy (MiniRocket-CNN, Kernel-Size)',
        # 'Mean-Val-Accuracy (Rocket-CNN, Kernel-Size)',
        'Mean-Val-Accuracy (MiniRocket-CNN, Num-Conv-Layers)',
        'Mean-Val-Accuracy (Rocket-CNN, Num-Conv-Layers)',
    ]

    for report_filename, category, title in zip(filenames, categories, titles):
        print(f'Processing {report_filename}...')
        report = pd.read_csv(os.path.join(report_path, report_filename), index_col=False)
        fig_filepath = os.path.join(report_path, report_filename.replace('summary', 'cd_diagram').replace('csv', 'png'))
        draw_cd_diagram(df_perf=report, title=title, 
                        category=category, sort_by=sort_by, metric=metric,
                        labels=True, filepath=fig_filepath)
        

def eval_plots():
    report_path = 'exp/optimal/analysis'
    filenames = [
        'eval_all.csv',
    ]

    categories = ['classifier_name']
    sort_by = 'dataset_name'
    metric = 'accuracy'

    titles = ['Mean Test Accuracy']

    for report_filename, category, title in zip(filenames, categories, titles):
        print(f'Processing {report_filename}...')
        report = pd.read_csv(os.path.join(report_path, report_filename), index_col=False)
        fig_filepath = os.path.join(report_path, report_filename.replace('all', 'cd_diagram').replace('csv', 'png'))
        draw_cd_diagram(df_perf=report, title=title, 
                        category=category, sort_by=sort_by, metric=metric,
                        labels=True, filepath=fig_filepath)
        

if __name__ == '__main__':
    # param_study_plots()
    eval_plots()
    