#!/usr/bin/python3

import numpy as np
import adi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

sample_rate = 1.0e6 # 1,500,000 Hz sample rate
center_freq = 96.5e6# 94.5e6 MHz frequency
fft_size    = 1024  # samples per FFT
rx_buf_size = fft_size # samples per PLUTO rx buffer

sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = sample_rate
#sdr.gain_control_mode = 'manual'
#sdr.rx_hardwaregain = 70.0 # dB
sdr.gain_control_mode_chan0 = 'slow_attack'
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = rx_buf_size # buffer size to store rcvd samples

init_vals = np.empty(fft_size)
init_vals.fill(0)

# the range of the FFT frequency bins
start_freq = center_freq - sample_rate/2.0
stop_freq = center_freq + sample_rate/2.0
step = (stop_freq - start_freq)/len(init_vals)

# square image
# first arg = num_rows,
# second arg = num_cols (fft_size)
img_width = fft_size
img_height = fft_size
waterfall_image = np.zeros((fft_size, fft_size))
for i in range(img_height):
    samples = sdr.rx()

    fft_res = np.fft.fftshift(np.fft.fft(samples))
    #samples_mag = np.abs(fft_res)
    waterfall_image[i,:] = np.abs(fft_res)

fig = plt.figure()

im = plt.imshow(waterfall_image, extent=[start_freq, stop_freq, start_freq, stop_freq], animated=True)

def updatefig(*args):
    for i in range(img_height):
        samples = sdr.rx()
    
        fft_res = np.fft.fftshift(np.fft.fft(samples))
        #samples_mag = np.abs(fft_res)
        waterfall_image[i,:] = np.abs(fft_res)

    im.set_array(waterfall_image)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

plt.show()

