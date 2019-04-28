import cv2 as cv
import numpy as np
import os
from PIL import Image

rootdir='/media/ustb/Dataset2/biovid/PartA/crop_front_img_10/0'
targetroot='E:/biovid/PartB/framecanny'

for subdir1 in os.listdir(rootdir):
    if (subdir1 != '.DS_Store'):
        subdir=rootdir+'/'+subdir1
        targetsubdir=targetroot+'/'+subdir1
        if not os.path.exists(targetsubdir):
            os.makedirs(targetsubdir)
        print(subdir)
        for filename1 in os.listdir(subdir):
            if (filename1 != '.DS_Store'):
                filename=subdir+'/'+filename1
                name=filename

                img=cv.imread(name)
                blurred = img


                # 灰度图像
                gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
                # 图像梯度
                xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
                ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
                # 计算边缘
                # 50和150参数必须符合1：3或者1：2
                edge_output = cv.Canny(xgrad, ygrad, 50, 150)
                cv.imshow('win', edge_output)
                cv.waitKey(0)
                    #edge_output=Image.fromarray(edge_output)
                    #edge_output.save(targetname)
