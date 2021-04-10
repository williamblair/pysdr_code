#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

Nsymbols = 1000

# generate BPSK IQ symbols with AWGN noise
x_symbols = np.random.randint(0, 2, Nsymbols)*2 - 1 # -1 and 1
noise = (np.random.randn(Nsymbols) + 1j*np.random.randn(Nsymbols))/np.sqrt(2) # AWGN with unity power
result = x_symbols + noise * np.sqrt(0.01) # noise power of 0.01
print(result)
plt.plot(np.real(result), np.imag(result), '.')
plt.grid(True)
plt.show()

# Save to a file
print(type(result[0])) # check data type (numpy.complex128)
result = result.astype(np.complex64) # 32bits I, 32bits Q
print(type(result[0]))
result.tofile('bpsk_noise.iq')

