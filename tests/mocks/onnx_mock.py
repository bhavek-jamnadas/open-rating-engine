"""Lightweight fake so ONNX scoring never needs a real model or onnxruntime.

``OnnxScoring._load_model`` returns an ``onnxruntime.InferenceSession``. We patch
that seam with :class:`FakeSession`, which declares a fixed set of named inputs
and returns a deterministic prediction (the row-wise sum of the fed feature
columns) so tests can assert on outputs without a real ONNX graph.
"""

import numpy as np


class _FakeInput:
    def __init__(self, name: str, type_: str = "tensor(float)"):
        self.name = name
        self.type = type_


class FakeSession:
    """Stand-in for ``onnxruntime.InferenceSession``."""

    def __init__(self, input_names):
        self._inputs = [_FakeInput(n) for n in input_names]

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feed):
        # feed: {name: ndarray[N, 1]}. Deterministic output = row-wise sum.
        arrays = [np.asarray(feed[i.name]).reshape(-1) for i in self._inputs]
        total = np.sum(arrays, axis=0) if arrays else np.array([])
        return [np.asarray(total).reshape(-1, 1)]


def make_fake_load_model(input_names):
    """Return a stand-in for ``OnnxScoring._load_model`` that ignores the path."""

    def _fake(onnx_path: str):
        return FakeSession(input_names)

    return _fake
