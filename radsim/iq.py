import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from fractions import Fraction

from . import bench
from . import signals


class RealToIQ(bench.Consumer, bench.Producer, bench.PlotableInstrument):
    _ax = None
    _signal = None
    _fs = None
    _f_baseband = None
    @dataclass
    class _Store:
        t: np.ndarray = np.array([], dtype=np.float64)
        iq: np.array = np.array([], dtype=np.complex128)
    _store = _Store()
    def __init__(self, f_baseband, fs):
        self._f_baseband = f_baseband
        self._fs = fs
    def get_consume_type(self):
        return signals.Real
    def get_product_type(self):
        return signals.IQ
    def consume(self, real):
        signal = scipy.signal.hilbert(real.signal) * np.exp(2 * np.pi * -self._f_baseband * real.t * 1j)
        f = Fraction(self._fs, real.fs)
        signal = scipy.signal.resample_poly(signal, f.numerator, f.denominator)
        t = np.linspace(0, len(signal) / self._fs, len(signal))
        self._store.t = np.concatenate((self._store.t, self._store.t[-1] + t)) if len(self._store.t) > 0 else t
        self._store.iq = np.concatenate((self._store.iq, signal))
        self._signal = signals.IQ(t, signal, self._f_baseband, self._fs)
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
        t = self._store.t
        s = self._store.iq
        self._ax[0].plot(t, s.real, label="real")
        self._ax[0].plot(t, s.imag, label="imag")
        self._ax[0].legend()
        self._ax[1].plot(t, s.real, s.imag)
        fft = np.roll(np.abs(np.fft.fft(s)), int(len(s) / 2))
        f_range = self._fs / 2
        self._ax[2].plot(np.linspace(-f_range + self._f_baseband, f_range + self._f_baseband, len(fft)), fft)

class IQToReal(bench.Consumer, bench.Producer, bench.PlotableInstrument):
    _ax = None
    _signal = None
    _fs = None
    _f_baseband = None
    def __init__(self, fs):
        self._fs = fs
    @dataclass
    class _Store:
        t: np.ndarray = np.array([], dtype=np.float64)
        signal: np.ndarray = np.array([], dtype=np.float64)
    _store = _Store()
    def get_consume_type(self):
        return signals.IQ
    def get_product_type(self):
        return signals.Real
    def consume(self, iq):
        self._f_baseband = iq.f_baseband
        f = Fraction(self._fs, iq.fs)
        signal = scipy.signal.resample_poly(iq.signal, f.numerator, f.denominator)
        t = np.linspace(0, len(signal) / self._fs, len(signal))
        i = signal.real * np.cos(2 * np.pi * iq.f_baseband * t)
        q = signal.imag * np.sin(2 * np.pi * iq.f_baseband * t)
        signal = i + q
        self._store.t = np.concatenate((self._store.t, self._store.t[-1] + t)) if len(self._store.t) > 0 else t
        self._store.signal = np.concatenate((self._store.signal, signal))
        self._signal = signals.Real(t, signal, self._fs)
    def produce(self):
        ret = self._signal
        self._signal = None
        return ret
    def n_figs(self):
        return 1
    def init_figs(self, figs):
        self._ax = []
        self._ax.append(figs[0].add_subplot(2, 1, 1))
        self._ax[0].set_xlabel("Time")
        self._ax[0].set_ylabel("Real Signal")
        self._ax[0].set_title("Real")
        self._ax.append(figs[0].add_subplot(2, 1, 2))
        self._ax[1].set_xlabel("Frequency")
        self._ax[1].set_ylabel("Real Signal")
        self._ax[1].set_title("Real")
        figs[0].tight_layout()
    def plot(self):
        t = self._store.t
        s = self._store.signal
        self._ax[0].plot(t, s)
        fft = np.roll(np.abs(np.fft.rfft(s)), int(len(s) / 2))
        f_range = self._fs / 2
        self._ax[1].plot(np.linspace(0, f_range, len(fft)), fft)
