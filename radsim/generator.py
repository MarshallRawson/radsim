import numpy as np

from . import signals
from . import bench

class Real(bench.Producer):
    _signal = None
    def __init__(self, t, signal, fs):
        self._signal = signals.Real(t, signal, fs)
    def get_product_type(self):
        return signals.Real
    def produce(self):
        ret = self._signal
        self._signal = None
        return ret

class IQ(bench.Producer):
    _signal = None
    def __init__(self, t, signal, f_baseband, fs):
        self._signal = signals.IQ(t, signal, f_baseband, fs)
    def get_product_type(self):
        return signals.IQ
    def produce(self):
        ret = self._signal
        self._signal = None
        return ret

