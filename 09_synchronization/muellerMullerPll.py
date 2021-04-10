#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Nsymbols = 100
SamplesPerSym = 8

bits = np.random.randint(0, 2, Nsymbols) # data to be transmitted

# generate the 'signal' with 8 samples per symbol (7 out of 8 samples are zeros)
# signal is BPSK so the 1/-1 is the I component, Q is always 0
x = np.array([])
for bit in bits:
    pulse = np.zeros(SamplesPerSym)
    pulse[0] = bit*2-1 # set first bit to 1 or -1
    x = np.concatenate((x, pulse)) # add the 8 samples to the signal

plt.figure(0)
plt.plot(x, '.-')
plt.grid(True)
plt.show()


# Create the raised cosine filter
# sample period = 1 second
#   symbol period is then 8 because there are 8 samples/symbol
# want time to be in the center, so time will go from -51 to +51
Ntaps = 101
beta = 0.35
Ts = SamplesPerSym # symbol period, assumes sample period is 1Hz (1 sample/second)
t = np.arange(-51, 52) # not inclusive of 52, so -51...+51
h = np.sinc(t/Ts) * np.cos(np.pi * beta * t/Ts) / (1 - (2*beta*t/Ts)**2) # make taps
plt.figure(1)
plt.plot(t, h, '.')
plt.grid(True)
plt.show()


# filter the signal
x_shaped = np.convolve(x, h) # our result pulse shaped samples
plt.figure(2)
plt.plot(x_shaped, '.-')
for i in range(Nsymbols):
    # // is integer division, so //2 is integer divide by 2
    plt.plot([i*SamplesPerSym + Ntaps//2 + 1, i*SamplesPerSym + Ntaps//2 + 1],
             [min(x_shaped), max(x_shaped)])
plt.grid(True)
plt.show()


# fractional delay filter to simulate time delay offset
# sinc in time domain = rectangle in frequency domain, so we don't remove any signal, just shift it for delay
'''
delay = 0.4 # fractional delay in samples
Ndelaytaps = 21
n = np.arange(Ndelaytaps) # 1,2,3,...
h = np.sinc(n - delay) # calculate filter taps
h *= np.hamming(Ndelaytaps) # window filter to ensure decay to 0 on both sides
h /= np.sum(h) # normalize to get unity gain
# apply delay simulation filter
samples = np.convolve(x_shaped, h)
# plot non-delayed vs delayed
plt.figure(3)
plt.plot(x_shaped, '.-')
plt.plot(samples, '.-')
plt.show()
'''

# apply a 13 kHz frequency offset
'''
Fs = 1e6 # sample rate of 1 MHz
Fo = 13000 # 13 KHz freq offset
Ts = 1/Fs # sample period
t = np.arange(0, Ts*len(samples), Ts) # time vector
freq_offset_samples = samples * np.exp(1j * 2 * np.pi * Fo * t) # perform freq shift (remember mult in time domain = shift in freq domain)

plt.figure(4)
plt.plot(samples*(1j+1), '.-')
plt.plot(freq_offset_samples, '.-')
plt.show()
'''

# mueller-muller clock recovery pll
samples = x_shaped # pulse shaped samples without any timing or freq offset
mu = 0
out = np.zeros(len(samples) + 10, dtype=np.complex)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # used to save 2 previous values plus current value
i_in = 0 # input sample index
i_out = 2 # output index; first two outputs are 0
while i_out < len(samples) and i_in < len(samples):
    out[i_out] = samples[i_in + int(mu)] # grab estimated "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out - 1])
    y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y - x)
    mu += SamplesPerSym + 0.3*mm_val
    i_in += int(np.floor(mu)) # round down to nearest int since it is being used as an index
    mu = mu - np.floor(mu) # remove integer part of mu
    i_out += 1 # increment output index
out = out[2:i_out] # remove first two and anything after i_out

plt.figure(3)
plt.plot(out, '.-')
plt.show()

