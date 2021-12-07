import numpy as np
import matplotlib.pyplot as plt

from . import plotable

class BipolarBitStream(plotable.Plotable):
    _signal = None
    _f_bit = None
    @classmethod
    def from_bytes(cls, byte_array, f_bit):
        if isinstance(byte_array, bytes): byte_array = np.frombuffer(byte_array, dtype=np.uint8)
        else: assert byte_array is np.ndarray and byte_array.ndim == 1 and byte_array.dtype == np.dtype('np.uint8')
        assert isinstance(f_bit, float) or isinstance(f_bit, int)
        bbs = cls()
        bbs._f_bit = f_bit
        bbs._signal = np.array(np.unpackbits(byte_array), dtype=np.int8) * 2 - 1
        return bbs
    def to_np_bytes(self):
        return np.packbits(np.array((self._signal + 1) / 2, dtype=np.uint8))
    def add_plot_to_figure(self, fig, subplot, name=None):
        name = "Bipolar Bit Stream" if name is None else name
        t = np.arange(0, len(self._signal) / self._f_bit, 1 / self._f_bit)
        t = np.ravel((t, t), order='F')[1:]
        y = np.ravel((self._signal, self._signal), order='F')[:-1]
        ax = fig.add_subplot(*subplot)
        ax.plot(t, y)
        ax.set_xlabel("Time")
        ax.set_ylabel("Bipolar Bits")
        ax.set_title(name)
        return self
    @classmethod
    def from_iq(cls, iq_signal, fs, f_baseband, f_carrier):
        t = np.arange(0, len(iq_signal) / fs, 1 / fs)
        signal = iq_signal * np.exp(2 * np.pi * (f_baseband - f_carrier) * t * 1j)
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.plot(t, signal.real, label='real')
        ax.plot(t, signal.imag, label='imag')
        ax.legend()
        
        ax = fig.add_subplot(2,1,2)
        ax.plot(t, np.angle(signal), label='phase')
        ax.plot(t, np.abs(signal), label='mag')
        ax.legend()
        plt.show()

class BPSK(plotable.Plotable):
    _f_bit = None
    _f_carrier = None
    _signal = None
    _t = None
    _fs = None
    @classmethod
    def from_bipolar_bit_stream(cls, bipolar_bit_stream, f_carrier, fs, p_carrier=0):
        assert isinstance(bipolar_bit_stream, BipolarBitStream)
        assert fs >= 2 * f_carrier
        assert (fs / f_carrier).is_integer()
        assert (fs / bipolar_bit_stream._f_bit).is_integer()
        assert (f_carrier / bipolar_bit_stream._f_bit).is_integer(), \
            "f_carrier / f_bit = " + str(f_carrier / bipolar_bit_stream._f_bit) + " is not an integer"
        bpsk = cls()
        bpsk._f_bit = bipolar_bit_stream._f_bit
        bpsk._f_carrier = f_carrier
        bpsk._fs = fs
        end_t = (len(bipolar_bit_stream._signal) - 1) / bpsk._f_bit
        bpsk._t = np.arange(0.0, end_t, 1/fs)
        carrier = np.sin(2 * np.pi * f_carrier * bpsk._t + p_carrier)
        samples_per_bit = int(fs / bpsk._f_bit)
        carrier = np.array_split(carrier, len(carrier) / samples_per_bit)
        bpsk._signal = np.hstack([samples * bipolar_bit_stream._signal[i] for i, samples in enumerate(carrier)])
        return bpsk
    def signal(self):
        return self._signal
    def add_plot_to_figure(self, fig, subplot, name=None):
        name = "BPSK" if name is None else name
        ax = fig.add_subplot(*subplot)
        ax.plot(self._t, self._signal.real)
        ax.set_xlabel("Time")
        ax.set_ylabel("BPSK Signal")
        ax.set_title(name)
        return self

def test_answer():
    b = b'\x37' * 5
    assert (BipolarBitStream.from_bytes(b, np.float64(10.0)).to_np_bytes() == np.array(np.frombuffer(b, dtype=np.uint8))).all()
