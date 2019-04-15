#-*- encoding: utf8 -*-
import os
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# 분석정보를 face_cascade에 저장을 한다.


while True:
    ret,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faceList = face_cascade.detectMultiScale(gray,1.5)
    for(x,y,w,h) in faceList:
        #for 문에서 리스트의 얼굴 좌표 객체들의 좌표값을 순서대로 받아옵니다
        cv.rectangle(frame,(x,y),(x+w),y+h),(0,255,0),3)
        #녹색으로 두깨 3의 사각형을 변수 frame의 영상데이터에 그림
    cv.imshow('image',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.realease()
cv.destoryAllWindows()



#C:\Users\이수진\PycharmProjects\DeepLearning_Practice\aaaa.py