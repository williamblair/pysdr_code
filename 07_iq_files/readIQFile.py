#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

samples = np.fromfile('bpsk_noise.iq', np.complex64)
print(samples)


# Plot constellation
plt.plot(np.real(samples), np.imag(samples), '.')
plt.grid(True)
plt.show()

