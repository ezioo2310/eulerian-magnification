import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("video_results/auto10-20Hz_2Lvl_50Amp_butterFilter.avi")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

position_vectors = {'X': [], 'Y': []}
while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()

    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    # import pdb; pdb.set_trace()

    #square_coords = {'x1':275, 'x2': 325, 'y1':500, 'y2':550}
    square_coords = {'x1':290, 'x2': 310, 'y1':510, 'y2':530}
    x1, x2, y1, y2 = square_coords['x1'], square_coords['x2'],square_coords['y1'],square_coords['y2']

    position_vectors['X'].append(np.mean(flow[x1:x2,y1:y2, 0]))
    position_vectors['Y'].append(np.mean(flow[x1:x2,y1:y2, 1]))

    # adds frame with cumulative movement arrow
    (h, w) = frame.shape[:2]
    start_X, start_Y = (500, 300)
    start_point = (start_X, start_Y)
    #end_point = (start_X + int(np.sum(position_vectors['X'])*30), start_Y + int(np.sum(position_vectors['Y'])*30))
    end_point = (start_X + int(np.sum(position_vectors['X'][-1:])*50), start_Y + int(np.sum(position_vectors['Y'][-1:])*50))
    color = (0, 255, 0)
    thickness = 2

    image = cv.arrowedLine(frame, start_point, end_point, color, thickness)
    cv.imshow('arrow', image) 

    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
