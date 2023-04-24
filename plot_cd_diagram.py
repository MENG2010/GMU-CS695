"""Plot CD diagram.

@author: Ying & Chuxiong
@email: y(dot)meng201011(at)gmail(dot)com
"""
import os
from pathlib import Path
import pandas as pd

from metric.cd_diagram import draw_cd_diagram


report_path = 'exp/eval'
report_filename = 'eval_all.csv'

report = pd.read_csv(os.path.join(report_path, report_filename), index_col=False)
draw_cd_diagram(df_perf=report, title='Accuracy', labels=True)