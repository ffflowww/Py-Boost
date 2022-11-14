"""Abstracts for the tree ensembles"""

import cupy as cp
import numpy as np

from .utils import pinned_array
from ..quantization.base import QuantileQuantizer, UniformQuantizer, UniquantQuantizer
import nvtx


class Ensemble:
    """
    Abstract class for tree ensembles.
    Contains prediction, importance and data transfer methods
    """

    @staticmethod
    def _default_postprocess_fn(x):
        return x

    def __init__(self):
        """Initialize ensemble"""
        self.models = None
        self.nfeats = None
        self.postprocess_fn = self._default_postprocess_fn
        self.base_score = None
        self._on_device = False

        self.quantization = 'Quanntile'
        self.quant_sample = 200000
        self.max_bin = 256
        self.min_data_in_bin = 3

    def to_device(self):
        """Move trained ensemble data to current GPU device

        Returns:

        """
        if not self._on_device:
            for tree in self.models:
                tree.to_device()
            self.base_score = cp.asarray(self.base_score)

            self._on_device = True

    def to_cpu(self):
        """Move trained ensemble data to CPU memory

        Returns:

        """
        if self._on_device:
            for tree in self.models:
                tree.to_cpu()
            self.base_score = self.base_score.get()

            self._on_device = False

    def __getstate__(self):
        """Get state dict on CPU for picking

        Returns:

        """
        self.to_cpu()
        return self.__dict__

    def quantize(self, X, eval_set):
        """Fit and quantize all sets

        Args:
            X: np.ndarray, train features
            eval_set: list of np.ndarrays, validation features

        Returns:

        """
        quantizer = self.quantization

        if type(quantizer) is str:

            params = {'sample': self.quant_sample, 'max_bin': self.max_bin, 'min_data_in_bin': self.min_data_in_bin,
                      'random_state': self.seed}

            if self.quantization == 'Quantile':
                quantizer = QuantileQuantizer(**params)
            elif self.quantization == 'Uniform':
                quantizer = UniformQuantizer(**params)
            elif self.quantization == 'Uniquant':
                quantizer = UniquantQuantizer(**params)
            else:
                raise ValueError('Unknown quantizer')

        X_enc = quantizer.fit_transform(X)
        eval_enc = [quantizer.transform(x['X']) for x in eval_set]

        return X_enc, quantizer.get_max_bin(), quantizer.get_borders(), eval_enc

    def predict_new(self, X, batch_size=100000):
        self.to_device()
        cur_dtype = np.float32
        n_streams = 2
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # result allocation
        n_out = 3
        cpu_pred_full = np.empty((X.shape[0], n_out), dtype=cur_dtype)
        # cpu_pred = pinned_array(np.array(self.base_score.get().tolist() * X.shape[0], dtype=cur_dtype))
        # gpu_pred = cp.array(cpu_pred)
        cpu_pred = [pinned_array(np.empty((batch_size, n_out), dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_pred = [cp.empty((batch_size, n_out), dtype=cur_dtype) for _ in range(n_streams)]
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=cur_dtype) for _ in range(n_streams)]


        # new tree format
        # if left/right node is negative it means that it shows index in values array
        names = ['feat', 'split_val', 'left_node', 'right_node']
        types = [np.int32, np.float32, np.int32, np.int32]
        custom_node = np.dtype({'names': names, 'formats': types})

        for n, tree in enumerate(self.models):
            new_format = np.zeros((tree.ngroups, tree.values.shape[0] - 1), dtype=custom_node)
            nf = np.zeros(new_format.shape[0] * new_format.shape[1] * 4, dtype=np.float32)

            # print(tree.feats.dtype, tree.val_splits.dtype, tree.split.dtype, tree.nans.dtype, tree.leaves.dtype, tree.values.dtype)
            # print(tree.feats.shape, tree.val_splits.shape, tree.split.shape, tree.nans.shape, tree.leaves.shape, tree.values.shape)
            # print(self.base_score)
            # print(self.base_score.shape)
            # for row in tree.feats:
            #     pass
                # pos = (row >= 0).sum()
                # neg = (row < 0).sum()
                # s = pos + neg
                # print(pos, neg, s)


            for i in range(new_format.shape[0]):
                for j in range(new_format.shape[1]):
                    assert tree.feats[i][j] >= 0
                    new_format[i][j]['feat'] = tree.feats[i][j]
                    new_format[i][j]['split_val'] = tree.val_splits[i][j]
                    new_format[i][j]['left_node'] = tree.split[i][j][0]
                    new_format[i][j]['right_node'] = tree.split[i][j][1]

                    ln, rn = new_format[i][j]['left_node'], new_format[i][j]['right_node']
                    if tree.feats[i][ln] < 0:
                        new_format[i][j]['left_node'] = -tree.leaves[ln][i]
                    if tree.feats[i][rn] < 0:
                        new_format[i][j]['right_node'] = -tree.leaves[rn][i]

            for h in range(3):
                for k in range(63):
                    nf[4 * (h * 63 + k)] = float(new_format[h][k]['feat'])
                    nf[4 * (h * 63 + k) + 1] = new_format[h][k]['split_val']
                    nf[4 * (h * 63 + k) + 2] = float(new_format[h][k]['left_node'])
                    nf[4 * (h * 63 + k) + 3] = float(new_format[h][k]['right_node'])

            tree.new_format = cp.array(nf, dtype=cp.float32)

        # print(X.shape)
        # print([i for i in range(0, X.shape[0], batch_size)])
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                with nvtx.annotate(f"pred: {k}"):
                    real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                    with nvtx.annotate(f"to_cpu"):
                        cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(cur_dtype)
                    stream.synchronize()
                    with nvtx.annotate(f"to_gpu"):
                        gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                    stream.synchronize()
                    with nvtx.annotate(f"base_score"):
                        gpu_pred[nst][:] = self.base_score
                    stream.synchronize()
                    with nvtx.annotate(f"calc_trees"):
                        for n, tree in enumerate(self.models):
                            tree.predict_from_new_kernel(gpu_batch[nst][:real_batch_len], gpu_pred[nst][:real_batch_len])
                            stream.synchronize()  # this one
                    stream.synchronize()
                    # print("!")
                    # print(real_batch_len)
                    with nvtx.annotate(f"post_proc"):
                        self.postprocess_fn(gpu_pred[nst][:real_batch_len]).get(out=cpu_pred[nst][:real_batch_len])

                    stream.synchronize()

                    with nvtx.annotate(f"copying"):
                        cpu_pred_full[i: i + real_batch_len] = cpu_pred[nst][:real_batch_len]
                    stream.synchronize()

        cp.cuda.get_current_stream().synchronize()
        return cpu_pred_full

    def predict(self, X, batch_size=100000):
        """Make prediction for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of float32, shape (n_data, n_outputs)
        """
        self.to_device()
        prediction = pinned_array(np.empty((X.shape[0], self.base_score.shape[0]), dtype=np.float32))

        n_streams = 2
        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(n_streams)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % n_streams] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                result = cp.zeros((x_batch.shape[0], self.base_score.shape[0]), dtype=np.float32)
                result[:] = self.base_score
                for n, tree in enumerate(self.models):
                    result += tree.predict(gpu_batch)

                self.postprocess_fn(result).get(stream=st, out=prediction[i: i + x_batch.shape[0]])

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return prediction

    def predict_leaves(self, X, iterations=None, batch_size=100000):
        """Predict tree leaf indices for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
            for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of uint32, shape (n_iterations, n_data, n_groups).
            For n_groups explanation check Tree class
        """
        if iterations is None:
            iterations = range(len(self.models))

        self.to_device()

        check_grp = np.unique([x.ngroups for x in self.models])
        if check_grp.shape[0] > 1:
            raise ValueError('Different number of groups in trees')

        ngroups = check_grp[0]
        leaves = pinned_array(np.empty((len(iterations), X.shape[0], ngroups), dtype=np.int32))

        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(2)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % 2] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                for j, n in enumerate(iterations):
                    self.models[n].predict_leaf(gpu_batch).get(stream=st, out=leaves[j, i: i + x_batch.shape[0]])

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return leaves

    def predict_staged(self, X, iterations=None, batch_size=100000):
        """Make prediction from different stages for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
            for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of float32, shape (n_iterations, n_data, n_out)
        """
        if iterations is None:
            iterations = list(range(len(self.models)))

        self.to_device()
        prediction = pinned_array(np.empty((len(iterations), X.shape[0], self.base_score.shape[0]), dtype=np.float32))

        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(2)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % 2] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                result = cp.zeros((x_batch.shape[0], self.base_score.shape[0]), dtype=np.float32)
                result[:] = self.base_score

                next_out = 0
                for n, tree in enumerate(self.models):
                    result += tree.predict(gpu_batch)
                    if n == iterations[next_out]:
                        self.postprocess_fn(result).get(
                            stream=st, out=prediction[next_out, i: i + x_batch.shape[0]]
                        )

                        next_out += 1
                        if next_out >= len(iterations):
                            break

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return prediction

    def get_feature_importance(self, imp_type='split'):
        """Get feature importance

        Args:
            imp_type: str, importance type, 'split' or 'gain'

        Returns:

        """
        self.to_cpu()

        assert imp_type in ['gain', 'split'], "Importance type should be 'gain' or 'split'"
        importance = np.zeros(self.nfeats, dtype=np.float32)

        for tree in self.models:
            sl = tree.feats >= 0
            acc_val = 1 if imp_type == 'split' else tree.gains[sl]
            np.add.at(importance, tree.feats[sl], acc_val)

        return importance
