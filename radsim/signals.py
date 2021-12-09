import numpy as np
from dataclasses import dataclass

@dataclass
class Bytes:
    data: np.ndarray = np.frombuffer(bytes(), dtype=np.uint8)

@dataclass
class Real:
    t: np.ndarray = np.array([], dtype=np.float64)
    signal: np.ndarray = np.array([], dtype=np.float64)
    fs: float = 0

@dataclass
class IQ:
    t: np.ndarray = np.array([], dtype=np.float64)
    signal: np.ndarray = np.array([], dtype=np.complex128)
    f_baseband: float = 0
    fs: float = 0

