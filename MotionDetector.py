# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import argparse
import os

# ///////////////////////// Motion and People Detector /////////////////////////
'''
    This algorithm detects motion and whether there are people in the video stream.
    Most of this algorithm was taken from Brandon Joffe's Motion Detection module in
    his Home Surveillance project. Big thanks to Brandon for putting this out there.

    I modified his Motion Detection module to work without needing a class object so that
    it could be more easily adapted to other uses, such as home automation.

    Besides tweaking a few parameters in his algorithm, I added the Brightness Reset logic,
    which will reset the background if a sudden light change occurs, like turning the lights
    on or off in a room.

    I also adapted the algorithm to take in a video source as an argument. The video source can
    be a local cam or a folder that contains a list of sequence images, as often found in research projects.
'''

# Configure argsparse variables
parser = argparse.ArgumentParser(description="Motion detector detects motion and puts boxes around people."
                                             "You must enter at least one of the following: camera index,"
                                             " path or url.")
parser.add_argument('-W', '--width',
                    default=640,
                    type=int,
                    help='Camera width')

parser.add_argument('-H', '--height',
                    default=480,
                    type=int,
                    help='Camera height')

parser.add_argument('-C', '--camera_index',
                    type=int,
                    help='Camera # index. Index starts with 0.')

parser.add_argument('-P', '--path',
                    type=str,
                    help='Path to folder that contains images')

parser.add_argument('-U', '--url',
                    type=str,
                    help='Web URL for webcam')

args = vars(parser.parse_args())
if args['url'] is None and args['path'] is None and args['camera_index'] is None:
    print("Enter a webcam URL, folder path, or camera index.")

# Define video source
Mode = None
if args.get('camera_index') is not None:
    camera = cv2.VideoCapture(args['camera_index'])
    Mode = 'local_camera'
elif args.get('path') is not None:
    # Convert path to cover Windows usage
    sep = os.path.sep
    if sep != '/':
        path = args['path'].replace(os.path.sep, '/')

    # Read image names in the given folder and adds them to a list.
    try:
        os.listdir(path)
    except FileNotFoundError:
        print("Path Not Found")
        exit()

    pic_stream = []
    for filename in os.listdir(path):
        pic_stream.append(filename)

    # Set mode to toggle proper handling at loop
    Mode = 'pic_stream'

    # Find out how many frames to run
    total_frames = len(pic_stream)
    if total_frames <= 0:
        print("No pictures found")
        exit()

# Init  variables for loop
frame_num = 0
history = 0
currentFrame = None
meanFrame = None
peopleRects = []
pic_stream_counter = 0

# Iterate through each of those items and perform motion detection
while True:
    if Mode is 'local_camera':
        ret, frame = camera.read()
    elif Mode is 'pic_stream':
        frame = cv2.imread(path + '/' + pic_stream[pic_stream_counter])
        total_frames -= 1
        pic_stream_counter += 1
        if total_frames == 0:
            print("End of picture stream.")
            exit(0)

    # Calculate mean standard deviation then determine if motion has actually occurred
    height, width, channels = frame.shape
    kernel = np.ones((5, 5), np.uint8)

    # Resize the frame, convert it to gray scale, filter and blur it
    #logging.debug('\n\n////////////////////// filtering 1 //////////////////////\n\n')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if meanFrame is None:
        meanFrame = gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #logging.debug('\n\n////////////////////// filtering 1.5 //////////////////////\n\n')
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 9)  # Filters out noise
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    #logging.debug('\n\n////////////////////// filtering 2 //////////////////////\n\n')
    # Initialise and build background model using frame averaging
    if history <= 3:  # Let the camera warm up
        currentFrame = gray
        history += 1
    elif history == 4:
        previousFrame = currentFrame
        currentFrame = gray
        meanFrame = cv2.addWeighted(previousFrame, 0.5, currentFrame, 0.5, 0)
        history += 1
    elif history == 5:
        previousFrame = meanFrame
        currentFrame = gray
        meanFrame = cv2.addWeighted(previousFrame, 0.5, currentFrame, 0.5, 0)
        # cv2.imwrite("avegrayfiltered.jpg", meanFrame)
        history += 1
    elif history > 2000 and len(peopleRects) == 0:  # Recalculate background model every 2000 frames only if there are no people in frame
        previousFrame = currentFrame
        currentFrame = gray
        history = 0

    #logging.debug('\n\n////////////////////// averaging complete //////////////////////\n\n')
    # Compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(meanFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Removes small holes i.e noise
    thresh = cv2.dilate(thresh, kernel, iterations=3)  # Increases white region by saturating blobs
    #cv2.imwrite("motion.jpg", thresh)
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #logging.debug('\n\n////////////////////// filtering & thresholding //////////////////////\n\n')
    peopleRects = []
    # Loop through all contours
    for c in contours:
        # If the contour is too small or too big, ignore it
        if cv2.contourArea(c) < 2000 or cv2.contourArea(c) > 90000:
            if cv2.contourArea(
                    c) > 100000:  # If it is ridiculously big reset background model it is likely that something is wrong
                history = 0
                break
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # Compute the bounding box for the contour
        # If the bounding box is equal to the width (made smaller never really covers whole width) 
        # or height of the frame it is likely that something is wrong - reset model
        if h == height or w >= width / 1.5:
            history = 0
            break

        if (h) > (w):
            occupied = True
            if (h) > (1.5 * w): # Most likely a person, this can be made more strict (average human ratio 5.9/1.6 = h/w = 3.6875)
                person = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            peopleRects.append(cv2.boundingRect(c))
    #logging.debug('\n\n////////////////////// Contour area done //////////////////////\n\n')
    history += 1

    # ('\n\n////////////////////// Brightness Reset //////////////////////\n\n'
    # Resets background if the standard deviation is too high, which would occur in sudden lighting changes
    if cv2.meanStdDev(thresh)[1] > 100:
        previousFrame = currentFrame
        currentFrame = gray
        meanFrame = currentFrame
        history = 0

    cv2.imshow("Frame", frame)
    cv2.imshow("Mean Frame", thresh)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
