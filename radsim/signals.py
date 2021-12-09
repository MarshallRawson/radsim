import numpy as np
from dataclasses import dataclass

@dataclass
class Bytes:
    data: np.ndarray

@dataclass
class Real:
    t: np.ndarray
    signal: np.ndarray
    fs: float = 0

@dataclass
class IQ:
    t: np.ndarray
    signal: np.ndarray
    f_baseband: int
    fs: int

