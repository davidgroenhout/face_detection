# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy
import pathlib
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, default='./opencv_face_detector_uint8.pb')
parser.add_argument(
    '--config', type=str, default='./opencv_face_detector.pbtxt')
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('--duration', type=float, default=10.0)
parser.add_argument('--tolerance', type=float, default=0.5)
args = parser.parse_args()
model = args.model
config = args.config
path = args.path
time_remaining = args.duration
tolerance = args.tolerance
count = 0
video = cv2.VideoCapture(0)
net = cv2.dnn.readNet(model, config)

while time_remaining > 0:
    try:
        ret, frame = video.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        output = net.forward()
        best = numpy.argmax(output[0, 0, :, 2])
        if output[0, 0, best, 2] > tolerance:
            (x1, y1, x2, y2) = (
                output[0, 0, 0, 3:7] * numpy.array([w, h, w, h])
                ).astype('int')
            current_time = time.time()
            cv2.imwrite(
                f'{path}\{current_time}.png',
                frame[y1:y2, x1:x2])
            print(f'Saved "{path}\{current_time}.png"')
            count = count + 1
        time.sleep(0.1)
        time_remaining = time_remaining - 0.1
    except Exception as e:
        print(e)
        break

video.release()
cv2.destroyAllWindows()