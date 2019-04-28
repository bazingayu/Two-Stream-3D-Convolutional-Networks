import cv2
from numpy import *
import pylab as pl
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
path="C:\\Software\\matlab2016\\toolbox\\vision\\visionutilities\\classifierdata\\cascade\\haar\\haarcascade_frontalface_alt2.xml"
name=[]
count_m=0
dir = 'E:\\biovid\\PartB\\video_class'
finalgabor=[]
if os.path.exists(dir):
    dirs=os.listdir(dir)
    for dirc in dirs:
        dirc1=dir+'\\'+dirc
        if os.path.exists(dirc1):
            dircs=os.listdir(dirc1)
            for dircc in dircs:
                filename=dirc1+'\\'+dircc
                count_m+=1
                print(count_m)
                savefilename='E:\\biovid\\PartB\\frame'+'\\'+dirc +'\\'+ 'FACE' + dircc
                savefilename = savefilename.rstrip(".mp4")
                isExists = os.path.exists(savefilename)
                # 判断结果
                if not isExists:
                    # 如果不存在则创建目录
                    os.makedirs(savefilename)
                #print(savefilename)
                count = 0
                capture = cv2.VideoCapture(filename)
                if capture.isOpened():
                    rval, im = capture.read()
                    count=count+1
                    frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    face_patterns = cv2.CascadeClassifier(path)
                    faces = face_patterns.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if faces is not ():
                        for x, y, z, w in faces:
                            roiImg = frame[y:y + w, x:x + z]
                            oldroiImg=roiImg
                    else:
                        roiImg=oldroiImg
                    m = np.shape(roiImg)
                    if (np.shape(m)[0] == 3):
                        oldroiImg1=roiImg
                    else:
                        roiImg=oldroiImg
                    roiImg = cv2.resize(roiImg, (300, 300), interpolation=cv2.INTER_CUBIC)
                    prev_gray = roiImg
                    while rval:
                        rval, im= capture.read()
                        count+=1
                        while rval:
                            while count%6==0:
                                frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                                face_patterns = cv2.CascadeClassifier(path)
                                faces = face_patterns.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                                if faces is not ():
                                    for x, y, z, w in faces:
                                        roiImg = frame[y:y + w, x:x + z]
                                        oldroiImg = roiImg
                                else:
                                    roiImg = oldroiImg
                                m = np.shape(roiImg)
                                if (np.shape(m)[0] == 3):
                                    oldroiImg1 = roiImg
                                else:
                                    roiImg = oldroiImg
                                roiImg = cv2.resize(roiImg, (300, 300), interpolation=cv2.INTER_CUBIC)
                                savefilename1=savefilename+'\\'+str(count)+'.jpg'
                                roiImg=Image.fromarray(roiImg)
                                roiImg.save(savefilename1)
                                rval, im = capture.read()
                                count += 1
                            else:
                                rval, im = capture.read()
                                count+=1
                        break