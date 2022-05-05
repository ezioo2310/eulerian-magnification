import numpy as np
import scipy
import scipy.signal as signal
import matplotlib.pyplot as plt

def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result


def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0, amplification_factor=1):
    print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    result *= amplification_factor
    return result

def butter_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, amplification_factor=1, order=5):
    omega = 0.5 * fps
    low = freq_min / omega
    high = freq_max / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    y *= amplification_factor
    return y

def plot_butter_filter(fps, freq_min, freq_max, order):
    omega = 0.5 * fps
    low = freq_min / omega
    high = freq_max / omega
    b, a = signal.butter(order, [low, high], btype='band')
    w, h = signal.freqz(b, a, fs=fps)
    plt.plot(w, abs(h), label="order = %d" % order)
    
def plot_orders_butter(fps, freq_min, freq_max):
    for order in [3,5,7]:
        plot_butter_filter(fps, freq_min, freq_max, order)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')    
    plt.show()