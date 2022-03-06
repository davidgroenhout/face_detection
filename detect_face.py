# -*- coding: utf-8 -*-
import cv2
import numpy

video = cv2.VideoCapture(0)
model = './opencv_face_detector_uint8.pb'
config = './opencv_face_detector.pbtxt'
net = cv2.dnn.readNet(model, config)

while True:
    try:
        ret, frame = video.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        output = net.forward()
        best = numpy.argmax(output[0, 0, :, 2])
        prob = output[0, 0, best, 2]
        colour = (0, 0, 255) if prob < 0.9 else (0, 255, 0)
        (x1, y1, x2, y2) = (
            output[0, 0, best, 3:7] * numpy.array([w, h, w, h])
            ).astype('int')
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour)
        cv2.putText(frame, prob.astype('str'), (x1, y1 - 10),
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