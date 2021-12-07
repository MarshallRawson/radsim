import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import plotable

class IQ(plotable.Plotable):
    _signal = None
    _fs = None
    _f_baseband = None
    _t = None
    @classmethod
    def from_real(cls, real_signal, fs, f_baseband):
        iq = cls()
        t = np.arange(0, len(real_signal) / fs, 1 / fs)
        iq._signal = scipy.signal.decimate(scipy.signal.hilbert(real_signal) * np.exp(2 * np.pi * -f_baseband * t * 1j), 2)
        iq._t = scipy.signal.decimate(t, 2)
        assert len(iq._signal) == len(real_signal) / 2
        iq._fs = fs / 2
        iq._f_baseband = f_baseband
        return iq
    def add_plot_to_figure(self, fig, subplots, name=None):
        name = "IQ" if name is None else name
        assert len(subplots) == 3
        ax = fig.add_subplot(*subplots[0])
        ax.plot(self._t, self._signal.real, label="real")
        ax.plot(self._t, self._signal.imag, label="imag")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("IQ Signal")
        ax.set_title(name)
        ax = fig.add_subplot(*subplots[1], projection='3d')
        ax.plot(self._t, self._signal.real, self._signal.imag)
        ax.set_xlabel("Time")
        ax.set_title(name + " 3D")
        ax = fig.add_subplot(*subplots[2])
        fft = np.roll(np.abs(np.fft.fft(self._signal)), int(len(self._signal) / 2))
        f_range = self._fs / 2
        ax.plot(np.linspace(-f_range + self._f_baseband, f_range + self._f_baseband, len(fft)), fft)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("IQ Signal")
        ax.set_title(name)
        return self

