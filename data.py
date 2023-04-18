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
MODULE = os.path.dirname(os.getcwd())
def load_dataset(name: str,
                 split: str=None,
                 return_X_y: bool=True,
                 return_type: str='numpy2d',
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
    downloaded_path = os.getcwd() if not downloaded_path else downloaded_path
    extracted_path = os.path.join(downloaded_path, name)
    print(f'[INFO] loading {name} dataset from {extracted_path}')
    return _load_dataset(name=name, 
                         split=split,
                         return_X_y=return_X_y,
                         return_type=return_type,
                         extract_path=downloaded_path,)
    

def test():
    """Test load_dataset function."""
    name = 'Beef'
    X, y = load_dataset(name, split='train')
    
    print(f'[TEST] loaded {len(y)} samples from {name} dataset.')
    print(f'[TEST] {X[10, :5]}; {y[:5]}')
    

if __name__ == '__main__':
    dataset_base = '../datasets'
    test()