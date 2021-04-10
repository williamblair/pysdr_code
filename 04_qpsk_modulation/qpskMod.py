#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

Nsymbols = 1000

x_int = np.random.randint(0, 4, Nsymbols) # x = 0...3 which is the integer combo of the binary values
x_degrees = x_int * 360/4.0 + 45 # 45, 135, 255, 315 degrees
x_radians = x_degrees * np.pi/180.0

# the result complex signal
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)

# add some noise
# AWGN (additive white gaussian noise) with unity power
n = (np.random.randn(Nsymbols) + 1j*np.random.randn(Nsymbols))/np.sqrt(2)
noise_power = 0.01
x_symbols = x_symbols + n * np.sqrt(noise_power)

# add phase noise (caused by local oscillator jitter)
phase_noise = np.random.randn(len(x_symbols)) * 0.1
x_symbols = x_symbols * np.exp(1j*phase_noise)

# plot the results
plt.figure(0)
plt.plot(np.real(x_symbols), np.imag(x_symbols), '.')
plt.xlabel("In Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()

