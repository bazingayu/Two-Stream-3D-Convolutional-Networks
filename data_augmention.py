import numpy as np
import argparse
import cv2
import os
import concurrent.futures
import glob


def transfer(subfiles):
    videos = np.load(subfiles)
    print(subfiles)
    print(np.shape(videos))
    s = subfiles.split('/')
    target_root1 = "/media/ustb/data1/yjw/dataset/iso_img_augment/" + s[8] + '/' + s[9]
    if not os.path.exists(target_root1):
        os.makedirs(target_root1)
    target_root = target_root1 + '/' + s[10][:-4]

    (num,m,n,z) = np.shape(videos)

    array1 = []
    array2 = []
    array3 = []
    array4 = []
    array5 = []
    array6 = []
    array7 = []
    array8 = []
    array9 = []

    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    array_7 = []
    array_8 = []
    array_9 = []

    array0 = []
    for i in range(num):
        image = np.array(videos[i] , dtype= np.uint8)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, -9, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_9.append(rotated)

        M = cv2.getRotationMatrix2D(center, -8, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_8.append(rotated)


        M = cv2.getRotationMatrix2D(center, -7, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_7.append(rotated)

        M = cv2.getRotationMatrix2D(center, -6, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_6.append(rotated)

        M = cv2.getRotationMatrix2D(center, -5, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_5.append(rotated)

        M = cv2.getRotationMatrix2D(center, -4, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_4.append(rotated)

        M = cv2.getRotationMatrix2D(center, -3, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_3.append(rotated)

        M = cv2.getRotationMatrix2D(center, -2, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_2.append(rotated)

        M = cv2.getRotationMatrix2D(center, -1, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array_1.append(rotated)

        M = cv2.getRotationMatrix2D(center, 1, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array1.append(rotated)

        M = cv2.getRotationMatrix2D(center, 2, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array2.append(rotated)


        M = cv2.getRotationMatrix2D(center, 3, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array3.append(rotated)

        M = cv2.getRotationMatrix2D(center, 4, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array4.append(rotated)

        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array5.append(rotated)

        M = cv2.getRotationMatrix2D(center, 6, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array6.append(rotated)

        M = cv2.getRotationMatrix2D(center, 7, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array7.append(rotated)

        M = cv2.getRotationMatrix2D(center, 8, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array8.append(rotated)

        M = cv2.getRotationMatrix2D(center, 9, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array9.append(rotated)

        M = cv2.getRotationMatrix2D(center, 0, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        array0.append(rotated)
    target_9 = target_root + '_19.npy'
    target_8 = target_root + '_18.npy'
    target_7 = target_root + '_17.npy'
    target_6 = target_root + '_16.npy'
    target_5 = target_root + '_15.npy'
    target_4 = target_root + '_14.npy'
    target_3 = target_root + '_13.npy'
    target_2 = target_root + '_12.npy'
    target_1 = target_root + '_11.npy'
    target1 = target_root + '_01.npy'
    target2 = target_root + '_02.npy'
    target3 = target_root + '_03.npy'
    target4 = target_root + '_04.npy'
    target5 = target_root + '_05.npy'
    target6 = target_root + '_06.npy'
    target7 = target_root + '_07.npy'
    target8 = target_root + '_08.npy'
    target9 = target_root + '_09.npy'

    target = target_root + '_00.npy'




    array_9 = np.array(array_9)
    array_8 = np.array(array_8)
    array_7 = np.array(array_7)
    array_6 = np.array(array_6)
    array_5 = np.array(array_5)
    array_4 = np.array(array_4)
    array_3 = np.array(array_3)
    array_2 = np.array(array_2)
    array_1 = np.array(array_1)
    array1 = np.array(array1)
    array2 = np.array(array2)
    array3 = np.array(array3)
    array4 = np.array(array4)
    array5 = np.array(array5)
    array6 = np.array(array6)
    array7 = np.array(array7)
    array8 = np.array(array8)
    array9 = np.array(array9)
    array0 = np.array(array0)


    np.save(target,array0)
    np.save(target1, array1)
    np.save(target2, array2)
    np.save(target3, array3)
    np.save(target4, array4)
    np.save(target5, array5)
    np.save(target6, array6)
    np.save(target7, array7)
    np.save(target8, array8)
    np.save(target9, array9)
    np.save(target_1, array_1)
    np.save(target_2, array_2)
    np.save(target_3, array_3)
    np.save(target_4, array_4)
    np.save(target_5, array_5)
    np.save(target_6, array_6)
    np.save(target_7, array_7)
    np.save(target_8, array_8)
    np.save(target_9, array_9)


root_dir = "/media/ustb/Dataset2/biovid/two_flow_data/iso/original_iso_10_100/"
for sub1 in os.listdir(root_dir):
    if sub1 == 'flow_train' or sub1 == 'flow_test' or sub1 == 'flow_val' :
        sub = root_dir + sub1
        for i in range(0, 5, 1):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                subfiles = glob.glob(sub + '/' + str(i) + "/*")
                count = 0
                for image_file in zip(subfiles, executor.map(transfer, subfiles)):
                    print(count)
                    count += 1
