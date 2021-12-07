#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import radsim.bpsk
import radsim.iq

real_fig = plt.figure(1)
n_plots = 2
fs = 16000
bit_rate = 500
bpsk_carrier = 1000
f_baseband = 50

bbs = radsim.bpsk.BipolarBitStream.from_bytes(b'\x37', bit_rate).add_plot_to_figure(real_fig, (2, 1, 1))

bpsk = radsim.bpsk.BPSK.from_bipolar_bit_stream(bbs, bpsk_carrier, fs, p_carrier=np.pi).add_plot_to_figure(real_fig, (2, 1, 2))

iq_fig = plt.figure(2)
iq = radsim.iq.IQ.from_real(bpsk.signal(), fs, f_baseband).add_plot_to_figure(iq_fig, ((3, 1, 1), (3, 1, 2), (3, 1, 3)))
#iq = radsim.iq.IQ.from_real(np.cos(40 * 2 * np.pi * np.arange(0, 1, 1/fs)), fs, f_baseband).add_plot_to_figure(iq_fig, ((3, 1, 1), (3, 1, 2), (3, 1, 3)))

bbs2 = radsim.bpsk.BipolarBitStream.from_iq(iq._signal, fs/2, f_baseband, bpsk_carrier)

plt.show()



