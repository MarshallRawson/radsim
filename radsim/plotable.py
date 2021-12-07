from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Plotable(ABC):
    _name = None
    def rename_plot(self, name):
        self._name = name
    @abstractmethod
    def add_plot_to_figure(self, fig, subplot, name=None):
        pass
