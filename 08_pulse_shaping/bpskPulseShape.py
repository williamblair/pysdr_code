#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Nsymbols = 10
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


