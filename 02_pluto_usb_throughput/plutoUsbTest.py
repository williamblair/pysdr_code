#!/usr/bin/python3
import numpy as np
import adi
import matplotlib.pyplot as plt
import time

sample_rate = 1.9e6 # 1,000,000 Hz sample rate
center_freq = 100e6 # 100,000,000 Hz frequency

sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = sample_rate
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)

rx_buf_size = 1024
sdr.rx_buffer_size = rx_buf_size # buffer size to store rcvd samples

# sample as fast as possible, 1024 samples at a time, 1000 times
total_samples = 0
start = time.time()
for i in range(5000):
    samples = sdr.rx()
    total_samples = total_samples + len(samples)
end = time.time()

seconds_passed = end - start
res_smpl_rate = total_samples / seconds_passed
expected_total_samples = sample_rate * seconds_passed

print('seconds passed: {0}'.format(seconds_passed))
print('desired samples rcvd: {0}'.format(expected_total_samples))
print('samples rcvd: {0}'.format(total_samples))
print('desired sample rate: {0} samples/sec'.format(sample_rate))
print('sample rate: {0} samples/sec'.format(res_smpl_rate))
#print(samples)
#print(len(samples))

