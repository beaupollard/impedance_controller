from scipy.signal import butter, lfilter, freqz
import numpy as np
import math
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    t=np.linspace(0,len(y)/fs,len(y))
    plt.rcParams.update({'font.size': 22})
    plt.plot(t,y,linewidth=4)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.show()
    return y