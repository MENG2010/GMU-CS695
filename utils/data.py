"""Load dataset by name.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

from sktime.datasets._data_io import (
    _load_dataset,
    _load_provided_dataset,
    load_tsf_to_dataframe,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies


# base path for datasets
MODULE = os.path.dirname('datasets')
def load_dataset(name: str,
                 split: str=None,
                 return_X_y: bool=True,
                 return_type: str='np3d',
                 downloaded_path: str=None,):
    """Load dataset by name.

    Args:
        name (str): name of the dataset to load.
        split (str, optional): `TRAIN`, `TEST`, or None to load both training and test dataset. Defaults to None.
        return_X_y (bool, optional): if True, returns (features, target) separately, otherwise returns a single dataframe. Defaults to True.
        return_type (str, optional): string represents a supported mtype. Defaults to None.

    Returns:
        X: sktime data container.
        y: 1D numpy array of target values.
    """
    downloaded_path = 'datasets' if not downloaded_path else downloaded_path
    extracted_path = os.path.join(downloaded_path, name)
    print(f'[INFO] loading {name} dataset from {extracted_path}')
    return _load_dataset(name=name, 
                         split=split,
                         return_X_y=return_X_y,
                         return_type=return_type,
                         extract_path=downloaded_path,)


def reshape(data, extractor=None):
    extractor = '' if not extractor else extractor.lower()

    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    num_samples, num_values = data.shape
    print(f'[TEST] type: {type(data)}; num_samples: {num_samples}; num_values: {num_values}')

    if extractor == 'rocket':
        num_features = 2
    elif extractor == 'minirocket':
        num_features = 4
    else:
        num_features = 1
        
    reshaped_data = np.zeros(shape=(num_samples, num_features, num_values//num_features))
    print(f'[INFO] reshape {data.shape} --> {reshaped_data.shape}')

    if extractor == 'rocket':
        for sample_id in range(num_samples):
            for val_id in range(num_values):
                # print(f'[TEST] ({sample_id}, {val_id}): {data[sample_id][val_id]}')
                if val_id % 2 == 0:
                    reshaped_data[sample_id][0][val_id//2] = data[sample_id][val_id]
                else:
                    reshaped_data[sample_id][1][val_id//2] = data[sample_id][val_id]
    elif extractor == 'minirocket':
        for sample_id in range(num_samples):
            for val_id in range(num_values):
                if val_id % 4 == 0:
                    reshaped_data[sample_id][0][val_id//4] = data[sample_id][val_id]
                elif val_id % 4 == 1:
                    reshaped_data[sample_id][1][val_id//4] = data[sample_id][val_id]
                elif val_id % 4 == 2:
                    reshaped_data[sample_id][2][val_id//4] = data[sample_id][val_id]
                else:
                    reshaped_data[sample_id][3][val_id//4] = data[sample_id][val_id]
    else:
        for sample_id in range(num_samples):
            for val_id in range(num_values):
                reshaped_data[sample_id][0][val_id] = data[sample_id][val_id]

    print(f'[INFO] After reshaped: {reshaped_data.shape}')
    return reshaped_data


def test():
    """Test load_dataset function."""
    name = 'Beef'
    X, y = load_dataset(name, split='train')
    
    print(f'[TEST] loaded {len(y)} samples from {name} dataset.')
    print(f'[TEST] {X[10, :5]}; {y[:5]}')
    

if __name__ == '__main__':
    dataset_base = '../datasets'
    test()