import tensorflow as tf


def inference(s_images, T_images, batch_size, n_classes, is_training,per):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    # one stream space
    with tf.device('/gpu:1'):
        with tf.variable_scope('s_conv1') as scope:
            weights1 = tf.get_variable('weights1',
                                      shape=[3, 3, 3, 3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases1 = tf.get_variable('biases1',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv1 = tf.nn.conv3d(s_images, weights1, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv1 = tf.nn.bias_add(s_conv1, biases1)
            s_conv1 = tf.contrib.layers.batch_norm(s_conv1,is_training = is_training)


            s_conv1 = tf.nn.relu(s_conv1, name=scope.name)
            s_conv1 = tf.nn.dropout(s_conv1, keep_prob=per)
        with tf.variable_scope('s_conv2') as scope:
            weights1 = tf.get_variable('weights1',
                                      shape=[3, 3, 3,3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases1 = tf.get_variable('biases1',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv2 = tf.nn.conv3d(s_conv1, weights1, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv2 = tf.nn.bias_add(s_conv2, biases1)
            s_conv2 = tf.contrib.layers.batch_norm(s_conv2, is_training = is_training)

            s_conv2 = tf.nn.relu(s_conv2, name=scope.name)
            s_conv2 = tf.nn.dropout(s_conv2, keep_prob=per)

        # pool1 and norm1
        with tf.variable_scope('s_pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool3d(s_conv2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                                     padding='VALID', name='s_pooling1')

            norm1 = pool1
    with tf.device('/gpu:1'):
        with tf.variable_scope('s_conv3') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv3 = tf.nn.conv3d(norm1, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv3 = tf.nn.bias_add(s_conv3, biases2)
            s_conv3 = tf.contrib.layers.batch_norm(s_conv3, is_training = is_training)


            s_conv3 = tf.nn.relu(s_conv3, name='s_conv3')
            s_conv3 = tf.nn.dropout(s_conv3, keep_prob=per)

        with tf.variable_scope('s_conv4') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 3, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv4 = tf.nn.conv3d(s_conv3, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv4 = tf.nn.bias_add(s_conv4, biases2)
            s_conv4 = tf.contrib.layers.batch_norm(s_conv4, is_training = is_training)


            s_conv4 = tf.nn.relu(s_conv4, name='s_conv4')
            s_conv4 = tf.nn.dropout(s_conv4, keep_prob=per)
        with tf.variable_scope('s_conv44') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 5, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv4 = tf.nn.conv3d(s_conv4, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv4 = tf.nn.bias_add(s_conv4, biases2)
            s_conv4 = tf.contrib.layers.batch_norm(s_conv4, is_training = is_training)


            s_conv4 = tf.nn.relu(s_conv4, name='s_conv4')
            s_conv4 = tf.nn.dropout(s_conv4, keep_prob=per)

        # pool2 and norm2
        with tf.variable_scope('s_pooling2_lrn') as scope:

            norm2 = s_conv4
            pool2 = tf.nn.max_pool3d(norm2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='VALID',name='s_pooling2')
    with tf.device('/gpu:1'):
        with tf.variable_scope('s_conv5') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 5, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv5 = tf.nn.conv3d(pool2, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv5 = tf.nn.bias_add(s_conv5, biases2)
            s_conv5 = tf.contrib.layers.batch_norm(s_conv5, is_training = is_training)


            s_conv5 = tf.nn.relu(s_conv5, name='s_conv5')
            s_conv5 = tf.nn.dropout(s_conv5, keep_prob=per)
        
        with tf.variable_scope('s_conv6') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 5, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv6 = tf.nn.conv3d(s_conv5, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv6 = tf.nn.bias_add(s_conv6, biases2)
            s_conv6 = tf.contrib.layers.batch_norm(s_conv6, is_training = is_training)


            s_conv6 = tf.nn.relu(s_conv6, name='s_conv6')
            s_conv6 = tf.nn.dropout(s_conv6, keep_prob=per)
        with tf.variable_scope('s_conv66') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv6 = tf.nn.conv3d(s_conv6, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv6 = tf.nn.bias_add(s_conv6, biases2)
            s_conv6 = tf.contrib.layers.batch_norm(s_conv6, is_training = is_training)


            s_conv6 = tf.nn.relu(s_conv6, name='s_conv6')
            s_conv6 = tf.nn.dropout(s_conv6, keep_prob=per)
        with tf.variable_scope('s_pooling3_lrn') as scope:

            norm3 = s_conv6
            pool3 = tf.nn.max_pool3d(norm3, ksize=[1, 2, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',name='s_pooling3')

        with tf.variable_scope('s_conv7') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv7 = tf.nn.conv3d(pool3, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv7 = tf.nn.bias_add(s_conv7, biases2)
            s_conv7 = tf.contrib.layers.batch_norm(s_conv7, is_training = is_training)


            s_conv7 = tf.nn.relu(s_conv7, name='s_conv7')
            s_conv7 = tf.nn.dropout(s_conv7, keep_prob=per)
        with tf.variable_scope('s_conv8') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv8 = tf.nn.conv3d(s_conv7, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv8 = tf.nn.bias_add(s_conv8, biases2)
            s_conv8 = tf.contrib.layers.batch_norm(s_conv8, is_training = is_training)


            s_conv8 = tf.nn.relu(s_conv8, name='s_conv8')
            s_conv8 = tf.nn.dropout(s_conv8, keep_prob=per)
        with tf.variable_scope('s_conv88') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 64, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv8 = tf.nn.conv3d(s_conv8, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv8 = tf.nn.bias_add(s_conv8, biases2)
            s_conv8 = tf.contrib.layers.batch_norm(s_conv8, is_training = is_training)


            s_conv8 = tf.nn.relu(s_conv8, name='s_conv8')
            s_conv8 = tf.nn.dropout(s_conv8, keep_prob=per)

        with tf.variable_scope('s_pooling4_lrn') as scope:

            norm4 = s_conv8
            pool4 = tf.nn.max_pool3d(norm4, ksize=[1, 3, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID',name='s_pooling3')
        
        with tf.variable_scope('s_conv9') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[2, 3, 3, 64, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv9 = tf.nn.conv3d(pool4, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv9 = tf.nn.bias_add(s_conv9, biases2)
            s_conv9 = tf.contrib.layers.batch_norm(s_conv9, is_training = is_training)


            s_conv9 = tf.nn.relu(s_conv9, name='s_conv9')
            s_conv9 = tf.nn.dropout(s_conv9, keep_prob=per)
        with tf.variable_scope('s_conv10') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[2, 3, 3, 64, 1280],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[1280],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            s_conv10 = tf.nn.conv3d(s_conv9, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            s_conv10 = tf.nn.bias_add(s_conv10, biases2)
            s_conv10 = tf.contrib.layers.batch_norm(s_conv10, is_training = is_training)


            s_conv10 = tf.nn.relu(s_conv10, name='s_conv10')
            s_conv10 = tf.nn.dropout(s_conv10, keep_prob=per)
        with tf.variable_scope('s_pooling5_lrn') as scope:

            norm5 = s_conv10
            pool5 = tf.nn.max_pool3d(norm5 , ksize=[1, 2, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='VALID',name='s_pooling5')

        # local3
        with tf.variable_scope('s_local3') as scope:

            reshape = tf.reshape(pool5, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            print(dim)
            weights3 = tf.get_variable('weights3',
                                      shape=[11520, 5120],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases3 = tf.get_variable('biases3',
                                     shape=[5120],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            layer = tf.nn.bias_add(tf.matmul(reshape,weights3),biases3)
            layer = tf.contrib.layers.batch_norm(layer,is_training = is_training)
            s_local3 = tf.nn.relu(layer, name=scope.name)

    with tf.device('/gpu:1'):
        with tf.variable_scope('T_conv1') as scope:
            weights4 = tf.get_variable('weights4',
                                      shape=[3, 3, 3, 3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases4 = tf.get_variable('biases4',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv1 = tf.nn.conv3d(T_images, weights4, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv1 = tf.nn.bias_add(T_conv1, biases4)
            T_conv1 = tf.contrib.layers.batch_norm(T_conv1, is_training = is_training)


            T_conv1 = tf.nn.relu(T_conv1, name=scope.name)
            T_conv1 = tf.nn.dropout(T_conv1, keep_prob=per)
        with tf.variable_scope('T_conv2') as scope:
            weights4 = tf.get_variable('weights4',
                                      shape=[3, 3, 3, 3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases4 = tf.get_variable('biases4',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv2 = tf.nn.conv3d(T_conv1, weights4, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv2 = tf.nn.bias_add(T_conv2, biases4)
            T_conv2 = tf.contrib.layers.batch_norm(T_conv2, is_training = is_training)


            T_conv2 = tf.nn.relu(T_conv2, name=scope.name)
            T_conv2 = tf.nn.dropout(T_conv2, keep_prob=per)

        # pool1 and norm1
        with tf.variable_scope('T_pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool3d(T_conv2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                                     padding='VALID', name='T_pooling1')
            norm1 = pool1

        # conv3
        with tf.variable_scope('T_conv3') as scope:
            weights5 = tf.get_variable('weights5',
                                      shape=[3, 3, 3, 3, 3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases5 = tf.get_variable('biases5',
                                     shape=[3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv3 = tf.nn.conv3d(norm1, weights5, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv3 = tf.nn.bias_add(T_conv3, biases5)
            T_conv3 = tf.contrib.layers.batch_norm(T_conv3, is_training = is_training)


            T_conv3 = tf.nn.relu(T_conv3, name='T_conv3')
            T_conv3 = tf.nn.dropout(T_conv3, keep_prob=per)
    with tf.device('/gpu:1'):
        with tf.variable_scope('T_conv4') as scope:
            weights5 = tf.get_variable('weights5',
                                      shape=[3, 3, 3, 3, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases5 = tf.get_variable('biases5',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            T_conv4 = tf.nn.conv3d(T_conv3, weights5, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv4 = tf.nn.bias_add(T_conv4, biases5)
            T_conv4 = tf.contrib.layers.batch_norm(T_conv4, is_training = is_training)


            T_conv4 = tf.nn.relu(T_conv4, name='T_conv4')
            T_conv4 = tf.nn.dropout(T_conv4, keep_prob=per)
        with tf.variable_scope('T_conv44') as scope:
            weights5 = tf.get_variable('weights5',
                                      shape=[3, 3, 3, 5, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))

            biases5 = tf.get_variable('biases5',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            T_conv4 = tf.nn.conv3d(T_conv4, weights5, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv4 = tf.nn.bias_add(T_conv4, biases5)
            T_conv4 = tf.contrib.layers.batch_norm(T_conv4, is_training = is_training)


            T_conv4 = tf.nn.relu(T_conv4, name='T_conv4')
            T_conv4 = tf.nn.dropout(T_conv4, keep_prob=per)


        # pool2 and norm2
        with tf.variable_scope('T_pooling2_lrn') as scope:

            norm2 = T_conv4
            pool2 = tf.nn.max_pool3d(norm2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                                     padding='VALID', name='T_pooling2')
    with tf.device('/gpu:1'):
        with tf.variable_scope('T_conv5') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 5, 5],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[5],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv5 = tf.nn.conv3d(pool2, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv5 = tf.nn.bias_add(T_conv5, biases2)
            T_conv5 = tf.contrib.layers.batch_norm(T_conv5, is_training = is_training)


            T_conv5 = tf.nn.relu(T_conv5, name='T_conv5')
            T_conv5 = tf.nn.dropout(T_conv5, keep_prob=per)
        
        with tf.variable_scope('T_conv6') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 5, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv6 = tf.nn.conv3d(T_conv5, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv6 = tf.nn.bias_add(T_conv6, biases2)
            T_conv6 = tf.contrib.layers.batch_norm(T_conv6, is_training = is_training)


            T_conv6 = tf.nn.relu(T_conv6, name='s_conv6')
            T_conv6 = tf.nn.dropout(T_conv6, keep_prob=per)
    with tf.device('/gpu:1'):
        with tf.variable_scope('T_conv66') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv6 = tf.nn.conv3d(T_conv6, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv6 = tf.nn.bias_add(T_conv6, biases2)
            T_conv6 = tf.contrib.layers.batch_norm(T_conv6, is_training = is_training)


            T_conv6 = tf.nn.relu(T_conv6, name='s_conv6')
            T_conv6 = tf.nn.dropout(T_conv6, keep_prob=per)
        with tf.variable_scope('T_pooling3_lrn') as scope:

            norm3 = T_conv6
            pool3 = tf.nn.max_pool3d(norm3, ksize=[1, 2, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',name='T_pooling3')

        with tf.variable_scope('T_conv7') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv7 = tf.nn.conv3d(pool3, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv7 = tf.nn.bias_add(T_conv7, biases2)
            T_conv7 = tf.contrib.layers.batch_norm(T_conv7, is_training = is_training)


            T_conv7 = tf.nn.relu(T_conv7, name='T_conv7')
            T_conv7 = tf.nn.dropout(T_conv7, keep_prob=per)
        with tf.variable_scope('T_conv8') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 16, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv8 = tf.nn.conv3d(T_conv7, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv8 = tf.nn.bias_add(T_conv8, biases2)
            T_conv8 = tf.contrib.layers.batch_norm(T_conv8, is_training = is_training)


            T_conv8 = tf.nn.relu(T_conv8, name='T_conv8')
            T_conv8 = tf.nn.dropout(T_conv8, keep_prob=per)

        with tf.variable_scope('T_conv88') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[3, 3, 3, 64, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv8 = tf.nn.conv3d(T_conv8, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv8 = tf.nn.bias_add(T_conv8, biases2)
            T_conv8 = tf.contrib.layers.batch_norm(T_conv8, is_training = is_training)


            T_conv8 = tf.nn.relu(T_conv8, name='T_conv8')
            T_conv8 = tf.nn.dropout(T_conv8, keep_prob=per)

        with tf.variable_scope('T_pooling4_lrn') as scope:

            norm4 = T_conv8
            pool4 = tf.nn.max_pool3d(norm4, ksize=[1, 3, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID',name='T_pooling4')
        
        with tf.variable_scope('T_conv9') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[2, 3, 3, 64, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv9 = tf.nn.conv3d(pool4, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv9 = tf.nn.bias_add(T_conv9, biases2)
            T_conv9 = tf.contrib.layers.batch_norm(T_conv9, is_training = is_training)


            T_conv9 = tf.nn.relu(T_conv9, name='T_conv9')
            T_conv9 = tf.nn.dropout(T_conv9, keep_prob=per)
    with tf.device('/gpu:1'):
        with tf.variable_scope('T_conv10') as scope:
            weights2 = tf.get_variable('weights2',
                                      shape=[2, 3, 3, 64, 1280],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases2 = tf.get_variable('biases2',
                                     shape=[1280],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            T_conv10 = tf.nn.conv3d(T_conv9, weights2, strides=[1, 1, 1, 1, 1], padding='SAME')
            T_conv10 = tf.nn.bias_add(T_conv10, biases2)
            T_conv10 = tf.contrib.layers.batch_norm(T_conv10, is_training = is_training)


            T_conv10 = tf.nn.relu(T_conv10, name='T_conv10')
            T_conv10 = tf.nn.dropout(T_conv10, keep_prob=per)
        with tf.variable_scope('s_pooling5_lrn') as scope:

            norm5 = T_conv10
            pool5 = tf.nn.max_pool3d(norm5 , ksize=[1, 2, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='VALID',name='T_pooling5')

        # local3
        with tf.variable_scope('T_local3') as scope:

            reshape = tf.reshape(pool5, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights6 = tf.get_variable('weights6',
                                      shape=[11520, 5120],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=None,
                                                                                  dtype=tf.float32))
            biases6 = tf.get_variable('biases6',
                                     shape=[5120],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            layer = tf.nn.bias_add(tf.matmul(reshape, weights6),biases6)
            layer = tf.contrib.layers.batch_norm(layer,is_training=is_training)
            T_local3 = tf.nn.relu(layer, name=scope.name)

        local3 = tf.concat([s_local3,T_local3],1)
    
        print(local3)

        # local4
        with tf.variable_scope('local4') as scope:
            weights7 = tf.get_variable('weights7',
                                      shape=[10240, 2048],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001, seed=None,
                                                                                  dtype=tf.float32))
            biases7 = tf.get_variable('biases7',
                                     shape=[2048],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.bias_add(tf.matmul(local3, weights7),biases7)
            local4 = tf.contrib.layers.batch_norm(local4,is_training = is_training)

            local4 = tf.nn.relu(local4)
            local4 = tf.nn.dropout(local4, keep_prob=per)
    

        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights10 = tf.get_variable('softmax_linear',
                                      shape=[2048, n_classes],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001, seed=None,
                                                                                  dtype=tf.float32))
            biases10 = tf.get_variable('biases10',
                                     shape=[n_classes],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights10), biases10, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (labels=labels, logits=logits, name='entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss',loss)
    return loss


# %%
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_steps = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_steps)
    return train_op


# %%
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy',accuracy)

    return accuracy
