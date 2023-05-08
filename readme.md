# GMU-CS695-Spring23
 
## Setup Development Environment

Install the following packages:
- [sktime](https://github.com/sktime/sktime#hourglass_flowing_sand-install-sktime) with deep-learning dependencies -- for TSC development.
- [networkx](https://pypi.org/project/networkx/) -- for CD-diagram analysis and plot.

## Parameter Study

Perform parameter study to find the optimal training configuration for individual datasets for each time series classifier. Scripts for parameter studies start with `paramstudy`.

## Evaluate TSCs

1. Evaluate datasets using the optimal TSCs learned from the `Parameter Study` section with scripts starting with `eval_optimal`.
2. Evaluate baseline TSCs using scripts `pipeline_eval_rocket-ridge.py`, `pipeline_eval_minirocket-ridge.py`, and `pipeline_eval_cnn.py`.

## Plot CD diagram

Use `plot_cd_diagram.py` to plot CD diagrams.


## Structure

1. All experimental results will be put in `exp` folder.
2. Results of parameter studies are in `exp/param_study`.
3. Results of evaluation with optimal configurations are in `exp/optimal`.