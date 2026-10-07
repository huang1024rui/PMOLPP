# PMOLPP demo

Run `demo.py` to reproduce the following six PMOLPP classification experiments. The package contains all required code and both datasets. ACC is computed during execution and printed to the console as a percentage and a fraction. No precomputed results are included, and the program does not save experiment results or logs.

| Dataset  | data_num | TSPC | Seeds     | eta_sig | eta_stop |       t |
| -------- | -------: | ---: | --------- | ------: | -------: | ------: |
| Spambase |        6 |  200 | 1225, 873 |    0.60 |     0.30 | 1000000 |
| Spambase |        6 |  400 | 9690, 108 |    0.50 |     0.30 |   10000 |
| GFE-2    |       15 |  250 | 10, 5253  |    0.75 |     0.50 |   10000 |

## Run

Tested with Python 3.9.12 and the versions in `requirements.txt`. From this directory in your selected Python environment:

```bash
python -m pip install -r requirements.txt
python demo.py
```

The default command runs all six cases. You can also run a dataset or a single case:

```bash
python demo.py --data_num 6
python demo.py --data_num 6 --tspc 200 --seed 1225
python demo.py --data_num 6 --tspc 200 --seed 873
python demo.py --data_num 6 --tspc 400 --seed 9690
python demo.py --data_num 6 --tspc 400 --seed 108
python demo.py --data_num 15 --tspc 250 --seed 10
python demo.py --data_num 15 --tspc 250 --seed 5253
```

`--data_num`, `--tspc`, and `--seed` filter the six cases listed above. `python demo.py --check_only` validates their data and configurations without fitting models. Windows users can run `run_demo.bat` with the same arguments from an activated Python environment.

## Protocol

TSPC denotes training samples per class. Splits are stratified with `random_state=seed`. Spambase features are divided by the training-set feature standard deviations (no centering); near-zero standard deviations are replaced by 1. GFE-2 is used without preprocessing. Both datasets retain the original class-contiguous sample ordering.

PMOLPP is fitted only on the training set, with the parameters listed above, `regularization=1e-8`, `max_layers=None`, graph neighbors `floor(sqrt(n_train))`, and `random_state=seed`. Test samples are transformed using the fitted model. ACC uses a 15-nearest-neighbor classifier fitted on the training embedding and labels. Numerical operations use one thread for reproducibility. Package versions and numerical platforms can affect floating-point computations.

The original GeoMLE helper may emit numerical warnings; the included PMOLPP implementation retains its existing matrix-rank fallback for non-finite estimates. The six examples correspond to the historical top two seeds by test ACC in each setting. They illustrate reproducibility of those specific trials and are not an unbiased repeated-random-split performance estimate.

## Files

- `demo.py`: experiment entry point.
- `run_demo.bat`: Windows launcher.
- `Data/`: Spambase and GFE-2 MAT data.
- `PMOLPP/`: algorithm, preprocessing and fixed parameter configuration.
- `GLPP_LE/geomle.py`: intrinsic-dimension estimation helper.
- `requirements.txt`: tested dependencies.
