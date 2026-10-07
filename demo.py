#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run six PMOLPP reviewer experiments and print their ACC to the console."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from PMOLPP import PMOLPP, PMOLPP_Preprocess, Data_num2PMOLPP_Parameters

ROOT = Path(__file__).resolve().parent
CASES = [(6, 200, 1225), (6, 200, 873), (6, 400, 9690),
         (6, 400, 108), (15, 250, 10), (15, 250, 5253)]
DATASETS = {6: ('Spambase', 2), 15: ('GFE2', 0)}


def load_data(data_num):
    name, data_type = DATASETS[data_num]
    data_path = ROOT / 'Data' / ('data%d_initial.mat' % data_num)
    mat = loadmat(data_path)
    x = np.asarray(mat[name + '_InitialVector'].T, dtype=float)
    n_classes = int(np.asarray(mat[name + '_NumOfClass']).item())
    per_class = int(np.asarray(mat[name + '_NumPerClass']).item())
    if x.ndim != 2 or len(x) != n_classes * per_class or not np.isfinite(x).all():
        raise ValueError('Invalid data shape or non-finite values: ' + str(data_path))
    y = np.repeat(np.arange(n_classes), per_class)
    return x, y, n_classes, per_class, data_type


def split_indices(y, per_class, tspc, seed):
    n_classes = len(np.unique(y))
    train, test = train_test_split(
        np.arange(len(y)), test_size=(per_class - tspc) * n_classes,
        random_state=seed, stratify=y)
    expected = np.full(n_classes, tspc)
    if not np.array_equal(np.bincount(y[train]), expected):
        raise ValueError('Training samples per class do not match TSPC.')
    return train, test


def fit_case(data_num, tspc, seed):
    x, y, _, per_class, data_type = load_data(data_num)
    train, test = split_indices(y, per_class, tspc, seed)
    parameters = Data_num2PMOLPP_Parameters(data_num, tspc)
    preprocess = PMOLPP_Preprocess(Data_type=data_type)
    x_train = preprocess.fit_transform(x[train])
    x_test = preprocess.transform(x[test])
    model = PMOLPP(random_state=seed, **parameters)
    z_train = model.fit_transform(x_train)
    z_test = model.transform(x_test)
    predicted = KNeighborsClassifier(n_neighbors=15).fit(z_train, y[train]).predict(z_test)
    return float(accuracy_score(y[test], predicted))



def selected_cases(args):
    cases = [c for c in CASES
             if (args.data_num is None or c[0] == args.data_num)
             and (args.tspc is None or c[1] == args.tspc)
             and (args.seed is None or c[2] == args.seed)]
    if not cases:
        raise ValueError('No matching case among the six specified reviewer experiments.')
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_num', type=int, choices=sorted(DATASETS))
    parser.add_argument('--tspc', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--check_only', action='store_true')
    args = parser.parse_args(argv)
    cases = selected_cases(args)
    if args.check_only:
        for data_num, tspc, seed in cases:
            x, y, _, per_class, _ = load_data(data_num)
            split_indices(y, per_class, tspc, seed)
            parameters = Data_num2PMOLPP_Parameters(data_num, tspc)
            print('%s TSPC=%d seed=%d; shape=%s; parameters=%s' % (
                DATASETS[data_num][0], tspc, seed, x.shape, parameters), flush=True)
        print('Input checks passed for %d cases. No models fitted.' % len(cases))
        return
    with threadpool_limits(limits=1):
        for data_num, tspc, seed in cases:
            print('\nDataset: %s, TSPC: %d, Seed: %d' % (
                DATASETS[data_num][0], tspc, seed), flush=True)
            acc = fit_case(data_num, tspc, seed)
            print('ACC: %.2f%% (%.16f)' % (100 * acc, acc), flush=True)
    print('\nCompleted %d cases.' % len(cases), flush=True)


if __name__ == '__main__':
    main()
