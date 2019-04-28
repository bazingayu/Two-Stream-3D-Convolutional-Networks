from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np

from PIL import Image
cwd1 = '/media/ustb/Dataset/CASME/rgb'
cwd2 = '/media/ustb/Dataset/CASME/flow/'


classes = os.listdir(cwd1)  # 人为 设定 5 类
for i in range(1, 27):
    writer = tf.python_io.TFRecordWriter("/media/ustb/Dataset/CASME/LOSO_test/test" + '_'+str(i)+".tfrecords")  # 要生成的文件

    for index, name in enumerate(classes):
        class_path1 = cwd1 + '/' + name + '/'
        class_path2 = cwd2 + '/' + name + '/'
        for img_name in os.listdir(class_path1):
            n = str(img_name[-6:-4])
            if int(n) != int(i):
            	continue
            else:
                img_path1 = class_path1 + img_name
                img_path2 = class_path2 + img_name
                img_raw1 = np.load(img_path1)
                img_raw1 = img_raw1.tostring()
                img_raw2 = np.load(img_path2)
                img_raw2 = img_raw2.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw1])),
                'img_raw2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw2]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
    print(str(i)+'is finished!')
    writer.close()

