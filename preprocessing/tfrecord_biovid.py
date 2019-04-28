from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import tensorflow as tf
import numpy as np

from PIL import Image
cwd1 = '/media/ustb/data1/yjw/dataset/iso_img_augment/pic_val/'
cwd2 = '/media/ustb/data1/yjw/dataset/iso_img_augment/pic_val/'

if not os.path.exists('/media/ustb/current/yjw/tfrecord/pic_pic_iso_augment_19'):
    os.mkdir('/media/ustb/current/yjw/tfrecord/pic_pic_iso_augment_19')
classes = [0,4]
writer = tf.python_io.TFRecordWriter("/media/ustb/current/yjw/tfrecord/pic_pic_iso_augment_19/val.tfrecords")  # 要生成的文件

files = []
for index, name in enumerate(classes):
    class_path1 = cwd1  + str(name) + '/'
    #class_paths.append(class_path1)
    for img_name in os.listdir(class_path1):
        file_ = str(name) + '/' + img_name
        #print(file_)
        files.append(file_)
random.shuffle(files)
for i in range(len(files)):
    print(i,len(files))
    img_path1 = cwd1 + files[i]
    img_path2 = cwd2 + files[i]
    s = files[i].split('/')
    index = int(s[0])
    if index == 4:
        index = 1
    img_raw1 = np.load(img_path1)
    print(np.shape(img_raw1))
    img_raw1 = img_raw1.tostring()
    img_raw2 = np.load(img_path2)
    img_raw2 = img_raw2.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'img_raw1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw1])),
        'img_raw2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw1]))
    }))  # example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  # 序列化为字符串
print('finished!')
writer.close()



