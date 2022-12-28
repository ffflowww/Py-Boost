import os
import nvtx
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import cupy as cp
from py_boost import GradientBoosting
import matplotlib.pyplot as plt
from py_boost.gpu.inference import EnsembleInference


def test_reg(target_splitter, batch_size, params):
    X, y = make_regression(params["x_size"], params["feat_size"], n_targets=params["y_size"], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    model = GradientBoosting('mse', 'r2_score',
                             ntrees=params["n_trees"], lr=.01, verbose=20, es=200, lambda_l2=1,
                             subsample=.8, colsample=.8, min_data_in_leaf=10, min_gain_to_split=0,
                             max_bin=256, max_depth=6, target_splitter=target_splitter)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test},])

    print("Testing orig prob...")
    with nvtx.annotate("pred orig prob"):
        model.predict_deprecated(X_test[:32*32], batch_size=batch_size)
    print("Testing orig...")
    with nvtx.annotate("pred orig"):
        yp_orig = model.predict_deprecated(X_test, batch_size=batch_size)

    print("Testing fast prob...")
    with nvtx.annotate("pred fast prob"):
        model.predict(X_test[:32*32], batch_size=batch_size)
    print("Testing fast...")
    with nvtx.annotate("pred fast"):
        yp_fast = model.predict(X_test, batch_size=batch_size)

    print("Creating new Inference class")
    with nvtx.annotate("new class creating"):
        inference = EnsembleInference(model)

    print("Testing inference all prob...")
    with nvtx.annotate("pred inference all prob"):
        inference.predict(X_test[:32 * 32], batch_size=batch_size)
    print("Testing inference all...")
    with nvtx.annotate("pred inference all"):
        yp_fast_all = inference.predict(X_test, batch_size=batch_size)

    diff = yp_orig - yp_fast
    diff2 = yp_orig - yp_fast_all
    diff3 = yp_fast - yp_fast_all
    print(f"Outs diff: {diff.sum()}")
    print(f"Outs diff2: {diff2.sum()}")
    print(f"Outs diff3: {diff3.sum()}")

    plt.plot(yp_orig - y_test)
    plt.savefig('error_orig.png')
    plt.clf()
    plt.plot(yp_fast - y_test)
    plt.savefig('error_fast.png')
    plt.clf()
    plt.plot(yp_fast_all - y_test)
    plt.savefig('error_fast_all.png')
    plt.clf()


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])

    params = {
        "x_size": 1050000,
        "feat_size": 50,
        "y_size": 3,
        "n_trees": 100
    }

    with nvtx.annotate("Test case 1, OneVsAll"):
        test_reg("OneVsAll", batch_size=100000, params=params)

    # with nvtx.annotate("Test case 2, Single"):
    #     test_reg("Single", batch_size=100000, params=params)

    print("Finish tests")

