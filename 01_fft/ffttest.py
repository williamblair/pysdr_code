#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# sample times array in seconds
# 0...100 in integer increments means we have a sample rate of 1 hz (1 sample/sec)
t = np.arange(100)

# The generated signal = 0.15 Hz 
s = np.sin(0.15*2*np.pi*t)

# apply windowing
# the FFT assumes an infinite signal and 'wraps around', so
# the goal is to make the very beginning and end of the sampled signal
# as close together as possible to avoid a spike in frequency.
#s = s * np.hamming(100)

# compute the fft of the signal with 100 samples
# fftshift rearranges the DC center component in the center,
# with -f/2 left, +f/2 right
S = np.fft.fftshift(np.fft.fft(s)) 
print(S)


# magnitude and phase angle of the frequencies
S_mag = np.abs(S)
S_phase = np.angle(S)

# frequency X axis. Since sample rate is 1Hz,
# the graph will be from -fs/2 ... +fs/2 = -0.5Hz to +0.5Hz,
# and since there are 100 samples the step is 1.0/100.0 Hz
f = np.arange(-0.5, 0.5, 1.0/100.0)

# graph the magnitude results
plt.figure(0)
plt.plot(f, S_mag, '.-')

# graph the phase angle results
plt.figure(1)
plt.plot(f, S_phase, '.-')

plt.show()

