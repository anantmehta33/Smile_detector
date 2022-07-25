#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:32:23 2022

@author: anantmehta
"""

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) #scaling factor has to be larger for smile,also change number of neighbours
        for (sx, sy, sw, sh) in smile: # For each detected smile:
            cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0,0,255), 2) 
    return frame
video_capture = cv2.VideoCapture(0) # We turn the webcam on.
while True: # We repeat infinitely (until break):
    _, frame = video_capture.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('video',frame) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.
video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
