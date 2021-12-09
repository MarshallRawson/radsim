import numpy as np

from . import bench
from . import signals

class Literal(bench.Producer):
    _data = None
    _block_size = None
    def __init__(self, byte_data, block_size=None):
        self._data = byte_data
        self._block_size = block_size
    def get_product_type(self):
        return signals.Bytes
    def produce(self):
        if len(self._data) == 0:
            return None
        elif self._block_size is None or self._block_size > len(self._data):
            ret = signals.Bytes(np.frombuffer(self._data, dtype=np.uint8))
            self._data = bytes()
            return ret
        else:
            ret = signals.Bytes(np.frombuffer(self._data[:self._block_size], dtype=np.uint8))
            self._data = self._data[self._block_size:]
            return ret
