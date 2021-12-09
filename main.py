#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import copy

import radsim.payload_bytes
import radsim.bpsk
import radsim.iq
import radsim.generator
import radsim.hackrf

fs = int(10e6)
f_baseband = int(26e6)
f_carrier = int(28e6)
f_bit = f_carrier / 20
universe_fs = f_carrier * 100

rx = +radsim.iq.RealToIQ(f_baseband, fs) >> +radsim.bpsk.IQDemodulator(f_bit, f_carrier)
print((
    radsim.payload_bytes.Literal(b'\x37\x37'*100, block_size=10)
    >> +radsim.bpsk.IQModulator(f_bit, f_baseband, f_carrier, fs)
    >> radsim.hackrf.Tx()
    #>> +radsim.bpsk.IQDemodulator(f_bit, f_carrier)
    #>> radsim.bpsk.RealModulator(f_bit, f_carrier, universe_fs)
    #>> rx
).run(plot=False))




