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

        self._new_format_created = False

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

    def create_new_format(self):
        # new tree format
        # the sign of the feat value shows the behaviour in case of nan
        # to the value written in feats an extra "1" is added to deal with zero
        # if left/right node is negative, it means that it shows index in values array (abs)
        # in case of negative value an extra "1" is added to deal with zero
        for n, tree in enumerate(self.models):
            n_gr = tree.ngroups
            gr_subtree_offsets = np.zeros(n_gr, dtype=np.int32)

            # memory allocation for new tree array
            total_size = 0
            for i in range(n_gr):
                total_size += (tree.feats[i] >= 0).sum()
                if i < n_gr - 1:
                    gr_subtree_offsets[i + 1] = total_size
            nf = np.zeros(total_size * 4, dtype=np.float32)

            # reformatting the tree
            for i in range(n_gr):
                q = [(0, 0)]
                while len(q) != 0:  # BFS
                    n_old, n_new = q[0]
                    if tree.nans[i][n_old] is False:
                        nf[4 * (gr_subtree_offsets[i] + n_new)] = float(tree.feats[i][n_old] + 1)
                    else:
                        nf[4 * (gr_subtree_offsets[i] + n_new)] = float(-(tree.feats[i][n_old] + 1))
                    nf[4 * (gr_subtree_offsets[i] + n_new) + 1] = float(tree.val_splits[i][n_old])
                    ln = tree.split[i][n_old][0]
                    rn = tree.split[i][n_old][1]

                    if tree.feats[i][ln] < 0:
                        nf[4 * (gr_subtree_offsets[i] + n_new) + 2] = float(-(tree.leaves[ln][i] + 1))
                    else:
                        new_node_number = q[-1][1] + 1
                        nf[4 * (gr_subtree_offsets[i] + n_new) + 2] = float(new_node_number)
                        q.append((ln, new_node_number))

                    if tree.feats[i][rn] < 0:
                        nf[4 * (gr_subtree_offsets[i] + n_new) + 3] = float(-(tree.leaves[rn][i] + 1))
                    else:
                        new_node_number = q[-1][1] + 1
                        nf[4 * (gr_subtree_offsets[i] + n_new) + 3] = float(new_node_number)
                        q.append((rn, new_node_number))
                    q.pop(0)

            tree.new_format = cp.array(nf, dtype=cp.float32)
            tree.new_format_offsets = cp.array(gr_subtree_offsets, dtype=cp.int32)

            # new arrays for output indexing
            ns = [0]
            ni = []
            gri = np.array(tree.group_index, dtype=np.int32)
            for gr_ind in range(tree.ngroups):
                ns.append((gri == gr_ind).sum() + ns[-1])
                for en, ind in enumerate(tree.group_index):
                    if ind == gr_ind:
                        ni.append(en)
            tree.new_out_sizes = cp.array(ns, dtype=cp.int32)
            tree.new_out_indexes = cp.array(ni, dtype=cp.int32)
        self._new_format_created = True

    def predict_new(self, X, batch_size=100000):
        self.to_device()
        if not self._new_format_created:
            self.create_new_format()

        cur_dtype = np.float32
        n_streams = 2  # don't change
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # result allocation
        n_out = self.base_score.shape[0]
        cpu_pred_full = np.empty((X.shape[0], n_out), dtype=cur_dtype)
        cpu_pred = [pinned_array(np.empty((batch_size, n_out), dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_pred = [cp.empty((batch_size, n_out), dtype=cur_dtype) for _ in range(n_streams)]

        # batch allocation
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=cur_dtype) for _ in range(n_streams)]

        cpu_batch_free_event = [None, None]
        cpu_out_ready_event = [None, None]
        last_batch_size = 0
        last_n_stream = 0
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                with nvtx.annotate(f"pred: {k}"):
                    real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                    with nvtx.annotate(f"to_cpu"):
                        if k >= 2:
                            cpu_batch_free_event[nst].synchronize()
                        cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(cur_dtype)

                    with nvtx.annotate(f"to_gpu"):
                        if k >= 2:
                            cpu_out_ready_event[nst].synchronize()
                        gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                        cpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                    with nvtx.annotate(f"base_score"):
                        gpu_pred[nst][:] = self.base_score

                    with nvtx.annotate(f"calc_trees"):
                        print(f"Batch size: {real_batch_len}, i: {i}")
                        for n, tree in enumerate(self.models):
                            tree.predict_fast(gpu_batch[nst][:real_batch_len], gpu_pred[nst][:real_batch_len])

                    with nvtx.annotate(f"copying"):
                        if k >= 2:
                            cpu_pred_full[i - 2 * batch_size: i - batch_size] = cpu_pred[nst][:batch_size]

                    with nvtx.annotate(f"post_proc"):
                        self.postprocess_fn(gpu_pred[nst][:real_batch_len]).get(out=cpu_pred[nst][:real_batch_len])
                        cpu_out_ready_event[nst] = stream.record(cp.cuda.Event(block=True))

                    last_batch_size = real_batch_len
                    last_n_stream = nst

        if int(np.floor(X.shape[0] / batch_size)) == 0:
            with nvtx.annotate(f"copying last1"):
                with map_streams[last_n_stream] as stream:
                    stream.synchronize()
                    cpu_pred_full[:last_batch_size] = cpu_pred[last_n_stream][:last_batch_size]
        else:
            with nvtx.annotate(f"copying last2"):
                with map_streams[1 - last_n_stream] as stream:
                    stream.synchronize()
                    cpu_pred_full[X.shape[0] - batch_size - last_batch_size: X.shape[0] - last_batch_size] = cpu_pred[1 - last_n_stream][:batch_size]
                with map_streams[last_n_stream] as stream:
                    stream.synchronize()
                    cpu_pred_full[X.shape[0] - last_batch_size:] = cpu_pred[last_n_stream][:last_batch_size]

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
