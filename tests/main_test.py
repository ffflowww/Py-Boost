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
                             max_bin=256, max_depth=6, target_splitter=target_splitter,
                             debug=False)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test},])

    # X_test.ravel()[np.random.choice(X_test.size, X_test.size // 10, replace=False)] = np.nan
    X_test[...] = np.nan
    print(X_test)

    print("Testing orig prob...")
    with nvtx.annotate("pred orig prob"):
        model._predict_deprecated(X_test[:32*32], batch_size=batch_size)
    print("Testing orig...")
    with nvtx.annotate("pred orig"):
        yp_orig = model._predict_deprecated(X_test, batch_size=batch_size)

    print("Testing fast prob...")
    with nvtx.annotate("pred fast prob"):
        model.predict(X_test[:32*32], batch_size=batch_size)
    print("Testing fast...")
    with nvtx.annotate("pred fast"):
        yp_fast = model.predict(X_test, batch_size=batch_size)

    print("Creating new Inference class")
    with nvtx.annotate("new class creating"):
        inference = EnsembleInference(model)

    print("Testing Inference class prob...")
    with nvtx.annotate("pred Inference class prob"):
        inference.predict(X_test[:32 * 32], batch_size=batch_size)
    print("Testing Inference class...")
    with nvtx.annotate("pred Inference class"):
        yp_fast_all = inference.predict(X_test, batch_size=batch_size)

    diff = abs(yp_orig - y_test)
    diff2 = abs(yp_fast - y_test)
    diff3 = abs(yp_fast_all - y_test)
    diff4 = abs(yp_fast_all - yp_fast)
    print(f"Outs diff_0: {diff.sum()}")
    print(f"Outs diff_1: {diff2.sum()}")
    print(f"Outs diff_2: {diff3.sum()}")
    print(f"Outs diff_Z: {diff4.sum()}")
    print(f"Outs diff_Zwothorig: {(yp_orig - yp_fast).sum()}")
    # print(y_test[0])
    # print(yp_orig[0])
    # print(yp_fast[0])
    # print(yp_fast_all[0])
    # print("-----")
    # print(y_test[500_000])
    # print(yp_orig[500_000])
    # print(yp_fast[500_000])
    # print(yp_fast_all[500_000])
    #
    plt.plot(yp_orig - y_test)
    plt.savefig('error_orig.png')
    plt.clf()
    plt.plot(yp_fast - y_test)
    plt.savefig('error_fast.png')
    plt.clf()
    plt.plot(yp_fast_all - y_test)
    plt.savefig('error_fast_all.png')
    plt.clf()
    plt.plot(yp_fast_all - yp_fast)
    plt.savefig('error_fast_vs_all.png')
    plt.clf()
    #
    #
    # print("Test staging")
    # stages = [5, 15, 20, 61, 99]
    # ps_orig = model._predict_staged_deprecated(X_test, iterations=stages, batch_size=100_000)
    # ps_new = model.predict_staged(X_test, iterations=stages, batch_size=100_000)
    #
    # print("Difference between regular predict and staged predict (should be zero):")
    # print("Old version:")
    # print((ps_orig[4] - yp_fast).sum())
    # print("New version:")
    # print((ps_new[4] - yp_fast).sum())
    # print("Shapes: (old, new)")
    # print(ps_orig.shape, ps_new.shape)
    #
    #
    # print("Test feature importance")
    # fi_orig_g = model._get_feature_importance_deprecated('gain')
    # fi_orig_s = model._get_feature_importance_deprecated('split')
    # fi_new_g = model.get_feature_importance('gain')
    # fi_new_s = model.get_feature_importance('split')
    # print("FI diff between old and new, should be zero")
    # print((fi_orig_s - fi_new_s).sum())
    # print((fi_orig_g - fi_new_g).sum())
    #
    # print("Test leaves predict")
    # l_orig = model._predict_leaves_deprecated(X_test, stages, batch_size=10000)
    # l_new = model.predict_leaves(X_test, stages, batch_size=10000)
    #
    # print("OLd and new, should be similar the first two for sure, the third could be different dur to old error")
    # print("@@@@@@@ 1")
    # print(l_orig[0][0])
    # print(l_new[0][0])
    # print("@@@@@@@ 2")
    # print(l_orig[0][77])
    # print(l_new[0][77])
    # print("@@@@@@@ 3")
    # print(l_orig[0][500_000])
    # print(l_new[0][500_000])
    # print("Shapes, (old, new)")
    # print(l_orig.shape, l_new.shape)
    # d = (l_orig - l_new).sum()
    # print(f"Diff: {d}")


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])

    params = {
        "x_size": 1050000,
        "feat_size": 50,
        "y_size": 16,
        "n_trees": 100
    }

    # with nvtx.annotate("Test case 1, OneVsAll"):
    #     test_reg("OneVsAll", batch_size=100000, params=params)

    with nvtx.annotate("Test case 2, Single"):
        test_reg("Single", batch_size=100000, params=params)

    print("Finish tests")

