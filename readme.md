# GMU-CS695-Spring23
 
## Setup Development Environment
Install [sktime](https://github.com/sktime/sktime#hourglass_flowing_sand-install-sktime).

## Structure

1. `configs` - base directory for configuration files.
2. `datasets` - base directory for all datasets.
3. `utils` - where util scripts (e.g., `data.py`) locate.
4. `dlrocket` - implementation of DL-Rocket.
5. `metric` - implementation of CD.
6. Each script starting with `task` (e.g., `task_download_datasets.py`) is a pipeline script for a particular task.


## (Optional) Download datasets
This is an optional one-time task. Downloading all datasets used for evaluation in this project via `task_download_datasets.py`.

```bash
python task_download_datasets.py
```

## Evaluate baseline TSCs
