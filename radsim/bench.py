from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from . import signals

class _Instrument(ABC):
    pass

class _Producer(_Instrument):
    @abstractmethod
    def produce(self):
        pass
    @abstractmethod
    def get_product_type(self):
        pass

class _Consumer(_Instrument):
    @abstractmethod
    def consume(self, data):
        pass
    @abstractmethod
    def get_consume_type(self):
        pass

class PlotableInstrument(_Instrument):
    _figs_enabled = False
    def __pos__(self):
        self._figs_enabled = True
        return self
    def __neg__(self):
        self._figs_enabled = False
        return self
    @abstractmethod
    def n_figs(self):
        pass
    @abstractmethod
    def init_figs(self, figs):
        pass
    @abstractmethod
    def plot(self):
        pass

class Bench:
    _figs_enabled = False
    _instruments = []
    @staticmethod
    def _check_types(left, right):
        assert left.get_product_type() == right.get_consume_type(), \
            str(left)+' produces '+str(left.get_product_type())+' != '+str(right)+' consumes '+str(right.get_consume_type())
    def __init__(self, left, right):
        if isinstance(left, _Instrument):
            assert isinstance(left, _Producer), 'Left instrument must be a producer!'
        if isinstance(right, _Instrument):
            assert isinstance(right, _Consumer), 'Right instrument must be a consumer!'
        if isinstance(left, _Instrument) and isinstance(right, _Instrument):
            Bench._check_types(left, right)
            self._instruments = [left, right]
        elif isinstance(left, _Instrument) and isinstance(right, Bench):
            if len(right._instruments) > 0: Bench._check_types(left, right._instruments[0])
            self._instruments = [left] + right._instruments
        elif isinstance(left, Bench) and isinstance(right, _Instrument):
            self._instruments = left._instruments
            if len(self._instruments) > 0: Bench._check_types(self._instruments[-1], right)
            self._instruments.append(right)
        elif isinstance(left, Bench) and isinstance(right, Bench):
            self._instruments = left._instruments
            if len(left._instruments) != 0 and len(left._instruments) != 0:
                Bench._check_types(left._instruments[-1], right._instruments[0])
            self._instruments += right._instruments
        else:
            assert False, str(left)+' and '+str(right)+' are not supported parameters'
    def __rshift__(self, rhs):
        return Bench(self, rhs)
    def __pos__(self):
        self._instruments = [+i for i in self._instruments]
        self._figs_enabled = True
        return self
    def __neg__(self):
        self._instruments = [-i for i in self._instruments]
        self._figs_enabled = False
        return self
    def run(self, plot=True):
        self._figs_enabled = plot
        assert self._instruments != [], 'Bench has not instruments to run!'
        assert not isinstance(self._instruments, _Consumer), 'Left most instrument must not be a consumer!'
        figs = []
        if self._figs_enabled:
            for instr in self._instruments:
                if isinstance(instr, PlotableInstrument) and instr._figs_enabled:
                    f = [plt.figure() for _ in range(instr.n_figs())]
                    instr.init_figs(f)
                    figs += f
        out_data = []
        while True:
            if len(self._instruments) == 0: break
            data = self._instruments[0].produce()
            if data is None:
                if isinstance(self._instruments[0], PlotableInstrument) and self._instruments[0]._figs_enabled and self._figs_enabled:
                    self._instruments[0].plot()
                self._instruments = self._instruments[1:]
                continue
            else:
                assert isinstance(data, self._instruments[0].get_product_type()), str(self._instruments[0]) + \
                    ' returned ' + str(type(data)) + ' which is not an instance of ' + str(self._instruments[0].get_product_type())
            for instr in self._instruments[1:-1]:
                instr.consume(data)
                data = instr.produce()
                assert data is not None, str(instr) + ' ended early'
                assert isinstance(data, instr.get_product_type()), str(instr) + \
                    ' returned ' + str(type(data)) + ' which is not an instance of ' + str(instr.get_product_type())
            self._instruments[-1].consume(data)
            if isinstance(self._instruments[-1], _Producer): out_data.append(self._instruments[-1].produce())
        if self._figs_enabled:
            plt.show()
        return out_data

class Instrument(_Instrument):
    def __rshift__(self, rhs):
        return Bench(self, rhs)
class Producer(_Producer, Instrument):
    pass
class Consumer(_Consumer, Instrument):
    pass

