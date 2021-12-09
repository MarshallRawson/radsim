import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate

from . import bench
from . import signals

#TODO: IQModulator and RealModulator share ALOT of code

class IQModulator(bench.Consumer, bench.Producer, bench.PlotableInstrument):
    @dataclass
    class _Store:
        bbs: np.ndarray = np.array([], dtype=np.int8)
        t: np.ndarray = np.array([], dtype=np.float64)
        signal: np.ndarray = np.array([], dtype=np.complex128)
    _store = _Store()
    _ax = None
    _f_bit = None
    _f_baseband = None
    _f_carrier = None
    _fs = None
    _p_carrier = None
    _signal = None
    _bipolar_bit_stream = None
    def __init__(self, f_bit, f_baseband, f_carrier, fs, p_carrier=0):
        self._f_bit = f_bit
        self._f_baseband = f_baseband
        self._f_carrier = f_carrier
        self._fs = fs
        self._p_carrier = p_carrier
    def get_consume_type(self):
        return signals.Bytes
    def get_product_type(self):
        return signals.IQ
    def consume(self, byte_array):
        bbs = np.array(np.unpackbits(byte_array.data), dtype=np.int8) * 2 - 1
        self._store.bbs = np.concatenate((self._store.bbs, bbs))
        self._bipolar_bit_stream = bbs
        end_t = (len(self._bipolar_bit_stream) - 1) / self._f_bit
        t = np.arange(0.0, end_t, 1/self._fs)
        carrier = np.exp((2 * np.pi * (self._f_carrier - self._f_baseband) * t * 1j)+ self._p_carrier)
        samples_per_bit = int(self._fs / self._f_bit)
        carrier = np.array_split(carrier, len(carrier) / samples_per_bit)
        signal = np.hstack([samples * self._bipolar_bit_stream[i] for i, samples in enumerate(carrier)])
        self._store.t = np.concatenate((self._store.t, self._store.t[-1] + t)) if len(self._store.t) > 0 else t
        self._store.signal = np.concatenate((self._store.signal, signal))
        self._signal = signals.IQ(t, signal, self._f_baseband, self._fs)
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
        self._ax[0].set_ylabel("Bipolar Bits")
        self._ax[0].set_title("Bipolar Bit Stream")
        self._ax.append(figs[0].add_subplot(2, 1, 2))
        self._ax[1].set_xlabel("Time")
        self._ax[1].set_ylabel("BPSK Signal")
        self._ax[1].set_title("BPSK")
        figs[0].tight_layout()
    def plot(self):
        t = np.arange(0, len(self._store.bbs) / self._f_bit, 1 / self._f_bit)
        t = np.ravel((t, t), order='F')[1:]
        y = np.ravel((self._store.bbs, self._store.bbs), order='F')[:-1]
        self._ax[0].plot(t, y)
        self._ax[1].plot(self._store.t, self._store.signal.real, label='real')
        self._ax[1].plot(self._store.t, self._store.signal.imag, label='imag')
        self._ax[1].legend()

class RealModulator(bench.Consumer, bench.Producer, bench.PlotableInstrument):
    @dataclass
    class _Store:
        bbs: np.ndarray = np.array([], dtype=np.int8)
        t: np.ndarray = np.array([], dtype=np.float64)
        signal: np.ndarray = np.array([], dtype=np.float64)
    _store = _Store()
    _ax = None
    _f_bit = None
    _f_carrier = None
    _fs = None
    _p_carrier = None
    _signal = None
    _bipolar_bit_stream = None
    def __init__(self, f_bit, f_carrier, fs, p_carrier=0):
        assert fs >= 2 * f_carrier
        assert (fs / f_carrier).is_integer()
        assert (fs / f_bit).is_integer()
        assert (f_carrier / f_bit).is_integer()
        self._f_bit = f_bit
        self._f_carrier = f_carrier
        self._fs = fs
        self._p_carrier = p_carrier
    def get_consume_type(self):
        return signals.Bytes
    def get_product_type(self):
        return signals.Real
    def consume(self, byte_array):
        bbs = np.array(np.unpackbits(byte_array.data), dtype=np.int8) * 2 - 1
        self._store.bbs = np.concatenate((self._store.bbs, bbs))
        self._bipolar_bit_stream = bbs
        end_t = (len(self._bipolar_bit_stream) - 1) / self._f_bit
        t = np.arange(0.0, end_t, 1/self._fs)
        carrier = np.sin(2 * np.pi * self._f_carrier * t + self._p_carrier)
        samples_per_bit = int(self._fs / self._f_bit)
        carrier = np.array_split(carrier, len(carrier) / samples_per_bit)
        signal = np.hstack([samples * self._bipolar_bit_stream[i] for i, samples in enumerate(carrier)])
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
        self._ax[0].set_ylabel("Bipolar Bits")
        self._ax[0].set_title("Bipolar Bit Stream")
        self._ax.append(figs[0].add_subplot(2, 1, 2))
        self._ax[1].set_xlabel("Time")
        self._ax[1].set_ylabel("BPSK Signal")
        self._ax[1].set_title("BPSK")
        figs[0].tight_layout()
    def plot(self):
        t = np.arange(0, len(self._store.bbs) / self._f_bit, 1 / self._f_bit)
        t = np.ravel((t, t), order='F')[1:]
        y = np.ravel((self._store.bbs, self._store.bbs), order='F')[:-1]
        self._ax[0].plot(t, y)
        self._ax[1].plot(self._store.t, self._store.signal)

class IQDemodulator(bench.Consumer, bench.PlotableInstrument):
    @dataclass
    class _Store:
        t: np.ndarray
        signal: np.ndarray
    _store = _Store(np.array([], dtype=np.float64), np.array([], dtype=np.complex128))
    _ax = None
    _f_bit = None
    _f_carrier = None
    def __init__(self, f_bit, f_carrier):
        self._f_bit = f_bit
        self._f_carrier = f_carrier
    def get_consume_type(self):
        return signals.IQ
    def get_product_type(self):
        return signals.Bytes
    def consume(self, iq):
        s = iq.signal * np.exp(2 * np.pi * (iq.f_baseband - self._f_carrier) * iq.t * 1j)
        t = iq.t
        self._store.t = np.concatenate((self._store.t, self._store.t[-1] + t)) if len(self._store.t) > 0 else t
        self._store.signal = np.concatenate((self._store.signal, s))
    def produce(self):
        return None
    def n_figs(self):
        return 1
    def init_figs(self, figs):
        self._ax = []
        self._ax.append(figs[0].add_subplot(2, 1, 1))
        self._ax[0].set_xlabel("Time")
        self._ax[0].set_ylabel("Complex signal")
        self._ax[0].set_title("Signal Shifted to " + str(self._f_carrier) + "Hz")
        figs[0].tight_layout()
    def plot(self):
        s = self._store.signal
        t = self._store.t
        mag = np.abs(s)
        phase = np.angle(s)
        self._ax[0].plot(t, mag, label='Magnitude')
        self._ax[0].plot(t, phase, label='Phase')
        self._ax[0].legend()
