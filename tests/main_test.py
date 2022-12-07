import os
import nvtx
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import cupy as cp
from py_boost import GradientBoosting


def test_reg(target_splitter, batch_size, params):
    X, y = make_regression(params["x_size"], params["feat_size"], n_targets=params["y_size"], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    model = GradientBoosting('mse', 'r2_score',
                             ntrees=params["n_trees"], lr=.01, verbose=20, es=200, lambda_l2=1,
                             subsample=.8, colsample=.8, min_data_in_leaf=10, min_gain_to_split=0,
                             max_bin=256, max_depth=6, target_splitter=target_splitter)
    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test},])

    print("Reformatting")
    with nvtx.annotate("reformatting"):
        model.create_new_format()

    print("Testing orig prob...")
    with nvtx.annotate("pred orig prob"):
        model.predict(X_test[:32*32], batch_size=batch_size)
    print("Testing orig...")
    with nvtx.annotate("pred orig"):
        yp_orig = model.predict(X_test, batch_size=batch_size)

    print("Testing fast prob...")
    with nvtx.annotate("pred fast prob"):
        model.predict_new(X_test[:32*32], batch_size=batch_size)
    print("Testing fast...")
    with nvtx.annotate("pred fast"):
        yp_fast = model.predict_new(X_test, batch_size=batch_size)

    diff = yp_orig - yp_fast
    print(f"Outs diff: {diff.sum()}")

    saved = False
    trouble_lines = []
    ok_lines = []
    for i, line in enumerate(diff):
        if line[0] != 0:
            if not saved:
                # np.savetxt("out.txt", diff[i-10: i+10000])
                saved = True
            trouble_lines.append(i)
        else:
            ok_lines.append(i)
    tl = np.array(trouble_lines, dtype=np.int32)
    ol = np.array(ok_lines, dtype=np.int32)
    # np.savetxt("trouble_lines.txt", tl)
    # np.savetxt("ok_lines.txt", ol)
    if len(trouble_lines) > 0:
        print(f"Troubles from: {tl[0]} till {tl[-1]}")
    else:
        print("All good")


if __name__ == '__main__':
    print(f"Start tests with cuda: {cp.cuda.runtime.runtimeGetVersion()}")
    print(os.environ['CONDA_DEFAULT_ENV'])

    params = {
        "x_size": 1050000,
        "feat_size": 50,
        "y_size": 16,
        "n_trees": 200
    }

    with nvtx.annotate("Test case 1, OneVsAll"):
        test_reg("OneVsAll", batch_size=100000, params=params)

    with nvtx.annotate("Test case 2, Single"):
        test_reg("Single", batch_size=100000, params=params)

    print("Finish tests")

