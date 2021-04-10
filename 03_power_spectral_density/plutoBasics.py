#!/usr/bin/python3
import numpy as np
import adi
import matplotlib.pyplot as plt
import time

sample_rate = 1.0e6 # 1,000,000 Hz sample rate
center_freq = 94.5e6 # 100,000,000 Hz frequency

sdr = adi.Pluto('ip:192.168.2.1')
#sdr.gain_control_mode = 'manual'
#sdr.rx_hardwaregain = 70.0 # dB
sdr.sample_rate = sample_rate
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)

rx_buf_size = 4096
sdr.rx_buffer_size = rx_buf_size # buffer size to store rcvd samples

samples = sdr.rx()
print('samples len: {0}'.format(len(samples)))
print('samples 0: {0}'.format(samples[0]))
print(samples)

# Power = average of magnitude squared
#       = (1/N) * sum(sqrt(i*i + q*q)**2)
#       = np.mean(np.abs(samples)**2)
#
# When signal has a mean of ~0, which is usually the case for SDRs,
# Power approximately equals variance
# = np.var(samples)
power_calc = np.mean(np.abs(samples)**2)
variance_calc = np.var(samples)
print(power_calc)
print(variance_calc)

# Power Spectral Density
# Alg:
# 1) take FFT of samples (result is complex floats)
# 2) take magnitude of the FFT (results is scalar floats)
# 3) Normalize by dividing by length of FFT
# 4) Square result to get power (power = magnitude**2)
# 5) Convert to dB using 10log_10()
# 6) Perform FFT shift to center result DC and have range -f/2 ... +f/2
N = 1024 # FFT length
samples = samples[0:N] # only use N samples
#samples = samples * np.hamming(len(samples)) # window the samples
PSD = (np.abs(np.fft.fft(samples)/N))**2
PSD_log = 10.0 * np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

# plot PSD results
# range is center frequency +- (sample_rate)/2
f = np.linspace(center_freq - sample_rate/2.0, center_freq + sample_rate/2.0, N)
plt.figure(0)
plt.plot(f, PSD_shifted)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.show()

