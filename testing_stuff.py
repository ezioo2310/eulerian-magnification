import eulerian_magnification as em

import cv2
import numpy as np
import scipy.fftpack
import scipy.signal
from matplotlib import pyplot

def show_frequencies(vid_data, fps, bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []
    averages_square = []
    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, :, :, :].sum())
            averages_square.append(vid_data[x,272-100:272+100,480-100:480+100,:].sum())
    averages = averages - min(averages)
    averages_square = averages_square - min(averages_square)

    charts_x = 1
    charts_y = 4
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("Pixel Average of the middle square")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages_square)
    
    import pdb; pdb.set_trace()
    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 3)
    pyplot.title("FFT of the whole image")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[len(freqs) // 2 + 1:]
    fft = fft[len(fft) // 2 + 1:]
    pyplot.plot(freqs, abs(fft))

    freqs = scipy.fftpack.fftfreq(len(averages_square), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages_square))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 4)
    pyplot.title("FFT of the square in the middle of the image")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[len(freqs) // 2 + 1:]
    fft = fft[len(fft) // 2 + 1:]
    pyplot.plot(freqs, abs(fft))
    pyplot.show()


vid, fps = em.load_video_float('baby.mp4')
img = vid[0,:,:,:]
start_point = (480-100,272-100)
end_point = (480+100,272+100)
img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
cv2.imshow('img',img)
cv2.waitKey()
show_frequencies(vid,fps)
# vid_res = em.eulerian_magnification(vid, fps, freq_min=0.6, freq_max=2, amplification=200, pyramid_levels=3, skip_levels_at_top=2)
# em.io.save_video(vid_res, fps, save_filename='baby_3_200.avi')
