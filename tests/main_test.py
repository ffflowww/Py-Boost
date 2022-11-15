import os

import nvtx

import joblib
from sklearn.datasets import make_regression
import numpy as np
import cupy as cp

from py_boost import GradientBoosting
import pickle


def test_reg(target_splitter, batch_size, pc=False):
    X, y = make_regression(2420000, 100, n_targets=3, random_state=42)
    if pc:
        X_test, y_test = X[:1920000], y[:1920000]
        trees = 200
        X, y = X[-500000:], y[-500000:]
    else:
        X_test, y_test = X[:192000], y[:192000]
        trees = 50
        X, y = X[-50000:], y[-50000:]
    model = GradientBoosting('mse', 'r2_score',
                             ntrees=trees, lr=.01, verbose=5, es=200, lambda_l2=1,
                             subsample=.8, colsample=.8, min_data_in_leaf=10, min_gain_to_split=0,
                             max_bin=256, max_depth=6, target_splitter=target_splitter)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test}, ])

    print("Testing orig...")
    with nvtx.annotate("pred orig"):
        yy = model.predict(X_test, batch_size=batch_size)
    # np.savetxt("out_orig.txt", yy)

    print("Testing new...")
    with nvtx.annotate("pred new"):
        yy2 = model.predict_new(X_test, batch_size=batch_size)
    # np.savetxt("out_new.txt", yy2)

    diff = yy - yy2
    print(f"Outs diff: {diff.sum()}")
    # np.savetxt("diff.txt", diff)


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])
    pc = True
    # pc = False


    with nvtx.annotate("Test case 1"):
        if pc:
            test_reg("OneVsAll", batch_size=320000, pc=pc)
        else:
            test_reg("OneVsAll", batch_size=8000, pc=pc)
        # test_reg("Single", batch_size=32000)

    print("Finish tests")
