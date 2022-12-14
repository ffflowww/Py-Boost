import os
import nvtx
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import cupy as cp
from py_boost import GradientBoosting
import matplotlib.pyplot as plt


def test_reg(target_splitter, batch_size, params):
    X, y = make_regression(params["x_size"], params["feat_size"], n_targets=params["y_size"], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    model = GradientBoosting('mse', 'r2_score',
                             ntrees=params["n_trees"], lr=.01, verbose=20, es=200, lambda_l2=1,
                             subsample=.8, colsample=.8, min_data_in_leaf=10, min_gain_to_split=0,
                             max_bin=256, max_depth=6, target_splitter=target_splitter)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test},])

    print("Testing orig...")
    with nvtx.annotate("pred orig"):
        yp_orig = model.predict(X_test, batch_size=batch_size)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_pred shape: {yp_orig.shape}")

    for i in range(0, X_test.shape[0], batch_size):
        print(f"Step: {i}")
        print(f"y_pred[{i}]: {yp_orig[i]}")

    print("Some troubles with last preds?")
    print(f"y_preds[800000:800005]:")
    print(yp_orig[800000:800005])

    print(f"\ny_preds[900000:900005]:")
    print(yp_orig[900000:900005])

    plt.plot(yp_orig - y_test)
    plt.savefig('error.png')


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])

    params = {
        "x_size": 1050000,
        "feat_size": 50,
        "y_size": 16,
        "n_trees": 100
    }

    with nvtx.annotate("Test case 1, OneVsAll"):
        test_reg("OneVsAll", batch_size=100000, params=params)

    # with nvtx.annotate("Test case 2, Single"):
    #     test_reg("Single", batch_size=100000, params=params)

    print("Finish tests")

