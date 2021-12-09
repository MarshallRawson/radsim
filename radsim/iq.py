import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

from . import bench
from . import signals

@dataclass
class _Store:
    t: np.ndarray = np.array([], dtype=np.float64)
    iq: np.array = np.array([], dtype=np.complex128)

class RealToIQ(bench.Consumer, bench.Producer, bench.PlotableInstrument):
    _ax = None
    _signal = None
    _f_baseband = None
    _signal_store = _Store()
    _fs = None
    _t = None
    def __init__(self, f_baseband):
        self._f_baseband = f_baseband
    def get_consume_type(self):
        return signals.Real
    def get_product_type(self):
        return signals.IQ
    def consume(self, real):
        signal = scipy.signal.decimate(scipy.signal.hilbert(real.signal) * np.exp(2 * np.pi * -self._f_baseband * real.t * 1j), 2)
        self._t = real.t[::2]
        self._fs = real.fs / 2
        self._signal_store.t = np.concatenate((self._signal_store.t, self._signal_store.t[-1] + self._t)) if len(self._signal_store.t) > 0 else self._t
        self._signal_store.iq = np.concatenate((self._signal_store.iq, signal))
        self._signal = signals.IQ(self._t, signal, self._fs)
        assert len(self._signal.signal) == len(real.signal) / 2
    def produce(self):
        ret = self._signal
        self._signal = None
        return ret
    def n_figs(self):
        return 1
    def init_figs(self, figs):
        self._ax = []
        self._ax.append(figs[0].add_subplot(3, 1, 1))
        self._ax[0].set_xlabel("Time")
        self._ax[0].set_ylabel("IQ Signal")
        self._ax[0].set_title("IQ")
        self._ax.append(figs[0].add_subplot(3, 1, 2, projection='3d'))
        self._ax[1].set_xlabel("Time")
        self._ax[1].set_ylabel("Real")
        self._ax[1].set_zlabel("Imag")
        self._ax[1].set_title("IQ 3D")
        self._ax.append(figs[0].add_subplot(3, 1, 3))
        self._ax[2].set_xlabel("Frequency")
        self._ax[2].set_ylabel("IQ Signal")
        self._ax[2].set_title("IQ")
        figs[0].tight_layout()
    def plot(self):
        t = self._signal_store.t
        s = self._signal_store.iq
        self._ax[0].plot(t, s.real, label="real")
        self._ax[0].plot(t, s.imag, label="imag")
        self._ax[0].legend()
        self._ax[1].plot(t, s.real, s.imag)
        fft = np.roll(np.abs(np.fft.fft(s)), int(len(s) / 2))
        f_range = self._fs / 2
        self._ax[2].plot(np.linspace(-f_range + self._f_baseband, f_range + self._f_baseband, len(fft)), fft)

