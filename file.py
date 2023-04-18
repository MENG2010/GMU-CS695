"""File I/O functions.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
import time
import numpy as np
import yaml
import json
import pandas as pd


def parse_yaml(filepath: str) -> dict:
    """Parse yaml file."""
    with open(filepath, 'r') as stream:
        try:
            contents = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(f'[Error] Failed to parse yaml file: {filepath}. \n{err}')
            
    return contents