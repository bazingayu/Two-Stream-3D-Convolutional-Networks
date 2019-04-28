import numpy as np
import argparse
import cv2
import os
import concurrent.futures
import glob


def transfer(subfiles):
    videos = np.load(subfiles)

    s = subfiles.split('/')

    target_root1 = "/media/ustb/data1/yjw/dataset/iso_img_augment/" + s[7] + '/' + s[8]
    if not os.path.exists(target_root1):
        os.makedirs(target_root1)
    target_filename = target_root1 + '/' + s[9]
    print(target_filename)
    if not os.path.exists(target_filename):
        flows = []
        for i in range(10):
            old_img1 = np.array(videos[i,:,:,:], dtype=np.uint8)
            new_img1 = np.array(videos[i+1,:,:,:], dtype=np.uint8)




            old_img = cv2.cvtColor(old_img1, cv2.COLOR_BGR2GRAY)
            new_img = cv2.cvtColor(new_img1, cv2.COLOR_BGR2GRAY)

            # 图像梯度
            xgrad = cv2.Sobel(old_img, cv2.CV_16SC1, 1, 0)
            ygrad = cv2.Sobel(old_img, cv2.CV_16SC1, 0, 1)
            # 计算边缘
            # 50和150参数必须符合1：3或者1：2
            old_img = cv2.Canny(xgrad, ygrad, 50, 150)

            # 图像梯度
            xgrad1 = cv2.Sobel(new_img, cv2.CV_16SC1, 1, 0)
            ygrad1 = cv2.Sobel(new_img, cv2.CV_16SC1, 0, 1)
            # 计算边缘
            # 50和150参数必须符合1：3或者1：2
            new_img = cv2.Canny(xgrad1, ygrad1, 50, 150)


            flow = cv2.calcOpticalFlowFarneback(old_img, new_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Use Hue, Saturation, Value colour model
            hsv = np.zeros(old_img1.shape, dtype=np.uint8)
            hsv[..., 1] = 255

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            rgb = rgb1.tolist()
            flows.append(rgb)

        flows1 = np.array(flows)
        np.save(target_filename,flows1)


root_dir = "/media/ustb/data1/yjw/dataset/iso_img_augment2/"
for sub1 in os.listdir(root_dir):
    if sub1 == 'canny_flow_train' or sub1 == 'canny_flow_val'or sub1 == 'canny_flow_test':
        sub = root_dir + sub1
        for i in range(0, 5, 1):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                subfiles = glob.glob(sub + '/' + str(i) + "/*")
                count = 0
                for image_file in zip(subfiles, executor.map(transfer, subfiles)):
                    print(count)
                    count += 1