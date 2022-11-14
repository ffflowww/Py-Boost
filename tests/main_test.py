import os

import nvtx

import joblib
from sklearn.datasets import make_regression
import numpy as np
import cupy as cp

from py_boost import GradientBoosting
import pickle


def test_reg(target_splitter, batch_size):
    X_, y_ = make_regression(1970000, 100, n_targets=3, random_state=42)
    # X_test, y_test = X[:1920000], y[:1920000]
    X, y = X_[-50000:], y_[-50000:]
    X_test, y_test = X_[:1920000], y_[:1920000]

    model = GradientBoosting('mse', 'r2_score',
                             ntrees=10, lr=.01, verbose=5, es=200, lambda_l2=1,
                             subsample=.8, colsample=.8, min_data_in_leaf=10, min_gain_to_split=0,
                             max_bin=256, max_depth=6, target_splitter=target_splitter)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test}, ])
    # with open("mod.model", 'wb') as outp:  # Overwrites any existing file.
    #     pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    # with open('mod.model', 'rb') as inp:
    #     model = pickle.load(inp)


    with nvtx.annotate("pred orig"):
        yy = model.predict(X_test, batch_size=batch_size)
    np.savetxt("out_orig.txt", yy)

    with nvtx.annotate("pred new"):
        yy2 = model.predict_new(X_test, batch_size=batch_size)
    np.savetxt("out_new.txt", yy2)

    diff = yy - yy2
    print(f"Outs diff: {diff.sum()}")
    np.savetxt("diff.txt", diff)


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])

    with nvtx.annotate("OTest case 1"):
        test_reg("OneVsAll", batch_size=320000)
        # test_reg("Single", batch_size=32000)

    print("Finish tests")
