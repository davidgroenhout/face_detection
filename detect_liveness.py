# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy

parser = argparse.ArgumentParser()
parser.add_argument(
    '--detect_model', type=str, default='./opencv_face_detector_uint8.pb')
parser.add_argument(
    '--detect_config', type=str, default='./opencv_face_detector.pbtxt')
parser.add_argument('--liveness_model', type=str, default='./liveness.pb')
parser.add_argument('--offset', type=float, default=0.0)
args = parser.parse_args()
detect_model = args.detect_model
detect_config = args.detect_config
liveness_model = args.liveness_model
video = cv2.VideoCapture(0)
detect_net = cv2.dnn.readNet(detect_model, detect_config)
liveness_net = cv2.dnn.readNet(liveness_model)

while True:
    try:
        ret, frame = video.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        detect_net.setInput(blob)
        output = detect_net.forward()
        best = numpy.argmax(output[0, 0, :, 2])
        (x1, y1, x2, y2) = (
            output[0, 0, best, 3:7] * numpy.array([w, h, w, h])
            ).astype('int')
        face = frame[y1:y2, x1:x2]
        face_blob = cv2.dnn.blobFromImage(face, 1, (256, 256), (104, 177, 123))
        liveness_net.setInput(face_blob)
        prob = liveness_net.forward()
        colour = (0, 0, 255) if prob[0, 0] < prob[0, 1] else (0, 255, 0)
        label = 'fake' if prob[0, 0] < prob[0, 1] else 'real'
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        cv2.imshow('Liveness detection', frame)
        key = cv2.waitKey(1)
        if key != -1:
            break
    except Exception as e:
        print(e)
        break

video.release()
cv2.destroyAllWindows()