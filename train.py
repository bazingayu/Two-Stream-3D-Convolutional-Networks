#-*-coding:utf-8-*-
import os
import numpy as np
import tensorflow as tf
import read
import model3 as model

N_CLASSES = 2
IMG_W = 256  # resize the image, if the input image is t oo large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 100
CAPACITY = 10000
MAX_STEP = 100000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate_base = 0.00001  # with current parameters, it is suggested to use learning rate<0.0001
#learning_rate = 0.00001
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
import shutil
import os

def run_training():
    shutil.rmtree('logs')
    os.mkdir('logs')
    shutil.rmtree('graphs')
    os.mkdir('graphs')
    # you need to change the directories to yours.
    logs_train_dir = './logs'
    s_filename = '/media/ustb/current/yjw/tfrecord/pic_pic_iso_augment_19/train.tfrecords'

    s_train_batch, T_train_batch, s_train_label_batch = read.inputs(s_filename, BATCH_SIZE, channel=3)
    x1 = tf.placeholder(dtype=tf.float32, shape=[None, 10, 100, 100, 3], name='img1')
    x2 = tf.placeholder(dtype=tf.float32, shape=[None, 10, 100, 100, 3], name='img2')
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y')
    is_training = tf.placeholder(tf.bool)
    #train_logits = model.inference(x1, x2, BATCH_SIZE, N_CLASSES, is_training)
    train_logits = model.inference(x1, x2, BATCH_SIZE, N_CLASSES, is_training,keep_prob)
    train_logits = tf.cast(train_logits, dtype=tf.float32)
    pre = tf.argmax(train_logits, 1)
    labels1 = tf.argmax(y, 1)
    labels1 = tf.cast(labels1, tf.int32)
    labels = y
    train_loss = model.losses(train_logits, labels1)
    global_steps = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step=global_steps,decay_steps=300,decay_rate=0.99)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(train_loss, global_step=global_steps)
    train_acc = model.evaluation(train_logits, labels)
    config1 = tf.ConfigProto(allow_soft_placement=True)
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    test_images1, test_images2, test_labels = read.inputs("/media/ustb/current/yjw/tfrecord/pic_cannyflow_iso_augment_19/test.tfrecords", BATCH_SIZE, channel=3)
    val_images1, val_images2, val_labels = read.inputs("/media/ustb/current/yjw/tfrecord/pic_cannyflow_iso_augment_19/val.tfrecords", BATCH_SIZE, channel=3)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()  # 保存操作
    train_writer = tf.summary.FileWriter("./graphs/train", sess.graph) #写入到的位置
    test_writer = tf.summary.FileWriter("./graphs/test", sess.graph) #写入到的位置
    val_writer = tf.summary.FileWriter("./graphs/val",sess.graph)
    merged = tf.summary.merge_all()
    print('start_training')
    max_ = 0
    # 开始训练过程
    for step in np.arange(MAX_STEP):
        img1, img2, label = sess.run([s_train_batch, T_train_batch, s_train_label_batch])
        feed_dict = {x1: img1, x2: img2, y: label, is_training:True,keep_prob:0.5}
        _, pre1 , tra_loss, tra_acc,  train_logits1, merged1,train_logits1 = sess.run([train_op, pre, train_loss, train_acc, train_logits,merged,train_logits],feed_dict=feed_dict)
        if step % 100 == 0 and step != 0 :
            test_image1, test_image2, label1 = sess.run([test_images1, test_images2, test_labels])
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            labelll = np.argmax(label,1)
            #print(train_logits1[1:10])
            labell = np.argmax(label1, 1)
            print(labelll)
            print(pre1)
            print(labell)

            feed_dict1 = {x1: test_image1, x2: test_image2, y: label1, is_training:True,keep_prob:1}
            [train_logits2,tra_acc1,pre1,merged2] = sess.run([train_logits,train_acc, pre ,merged], feed_dict=feed_dict1)
            print(pre1)
            #print(train_logits2[1:10])
            if (tra_acc1 > max_):
                max_ = tra_acc1
            print('test accuracy = %.2f%%, max_acc = %.2f%%' % (tra_acc1 * 100.0, max_ * 100.0))
            # 运行汇总操作，写入汇总
            train_writer.add_summary(merged1, step)
            test_writer.add_summary(merged2,step)

        if step % 100 == 0 or (step + 1) == MAX_STEP and step != 0:
            val_image1, val_image2, label1 = sess.run([val_images1, val_images2, val_labels])
            feed_dict1 = {x1: val_image1, x2: val_image2, y: label1, is_training:True,keep_prob:0.5}
            [tra_acc1,merged3] = sess.run([train_acc,merged], feed_dict=feed_dict1)
            print('val_accuracy =',tra_acc1)
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            val_writer.add_summary(merged3,step)

    coord.request_stop()
    coord.join(threads)
    sess.close()


run_training()
