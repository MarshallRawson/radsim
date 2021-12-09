#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import radsim.payload_bytes
import radsim.bpsk
import radsim.iq

fs = 10000
f_bit = 1000
f_bpsk_carrier = 1000
f_baseband = 50


rx = radsim.iq.RealToIQ(f_baseband) >> +radsim.bpsk.Demodulator(f_bit, f_bpsk_carrier)

(
    radsim.payload_bytes.Literal(b'\x37\x37', block_size=2)
    >> +radsim.bpsk.Modulator(f_bit, f_bpsk_carrier, fs)
    >> +rx
).run(plot=True)
