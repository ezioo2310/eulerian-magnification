import eulerian_magnification as em
from eulerian_magnification.pyramid import create_laplacian_image_pyramid
import cv2
import numpy as np
import scipy.fftpack
import scipy.signal
from matplotlib import pyplot

def plot_laplacian_pyramid(image, pyramid_levels):
    pyr = create_laplacian_image_pyramid(image, pyramid_levels)
    charts_x = 1
    charts_y = pyramid_levels
    pyplot.figure(figsize=(20, 10))
    # pyplot.subplots_adjust(vspace=.7)

    for i in range(pyramid_levels):
        pyplot.subplot(charts_x, charts_y, i+1)
        pyplot.xlabel(f"Level {i+1}")
        pyr[i] = pyr[i] - np.min(pyr[i])
        img = cv2.cvtColor(pyr[i].astype('float32'), cv2.COLOR_BGR2GRAY)
        img *=255
        img = img.astype(np.uint8)
        equ = cv2.equalizeHist(img)
        res = np.vstack((img,equ))
        pyplot.imshow(res,cmap='gray', vmin=0, vmax=255)
    pyplot.show()

def show_frequencies(vid_data, fps, square_coords,bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []
    averages_square = []
    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, :, :, :].sum())
            averages_square.append(vid_data[x,
                square_coords['x1']:square_coords['x2'],
                square_coords['y1']:square_coords['y2'],
                :].sum())
                
    averages = averages[:700]
    averages_square = averages_square[:700]
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

from eulerian_magnification.pyramid import create_laplacian_video_pyramid, collapse_laplacian_video_pyramid,create_gaussian_video_pyramid
from eulerian_magnification.transforms import temporal_bandpass_filter


def eulerian_magnification(vid_data, fps, freq_min, freq_max, amplification, pyramid_levels=4, skip_levels_at_top=2):
    vid_pyramid = create_laplacian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
    for i, vid in enumerate(vid_pyramid):
        if i < skip_levels_at_top or i >= len(vid_pyramid) - 1:
            continue
        bandpassed = temporal_bandpass_filter(vid, fps, freq_min=freq_min, freq_max=freq_max, amplification_factor=amplification)
        vid_pyramid[i] += bandpassed

    vid_data = collapse_laplacian_video_pyramid(vid_pyramid)
    return vid_data

vid, fps = em.load_video_float('baby.mp4')
square_coords = {'x1':272-75, 'x2': 272, 'y1':480-50, 'y2':480+50}
img = vid[0,:,:,:]
start_point = (square_coords['y1'],square_coords['x1'])
end_point = (square_coords['y2'],square_coords['x2'])
plot_laplacian_pyramid(img, pyramid_levels=4)
# img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
# cv2.imshow('img',img)
# cv2.waitKey()
# show_frequencies(vid,fps,square_coords)

# vid_res = eulerian_magnification(vid, fps, freq_min=0.6, freq_max=2, amplification=5, pyramid_levels=4, skip_levels_at_top=1)
# em.io.save_video(vid_res, fps, save_filename='baby_trialG2.avi')


### audi
# vid, fps = em.io._load_video('auto.mov') 
# square_coords = {'x1':100, 'x2': 300, 'y1':300, 'y2':500}
# # img = vid[0,:,:,:]
# start_point = (square_coords['y1'],square_coords['x1'])
# end_point = (square_coords['y2'],square_coords['x2'])
# # img = img[500:,1000:]
# # img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
# # cv2.imshow('img',img)
# # cv2.waitKey()

# vid=vid[:,500:,1000:,:]
# show_frequencies(vid,fps,square_coords)
