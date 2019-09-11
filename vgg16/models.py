
import numpy as np
import tensorflow as tf


def conv2d(inputs, filters, name, kernel_size = 3, strides = 1, padding='same', bias=False,
           dilation_rate = 1, trainable = True, activation = None, reuse = False):
    return tf.layers.conv2d(inputs, filters = filters,
                            kernel_size = kernel_size,
                            padding = padding,
                            strides = strides,
                            dilation_rate = dilation_rate,
                            activation = activation,
                            trainable = trainable,
                            reuse = reuse,
                            use_bias = bias,
                            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                            name = name)

def fullyConnect(inputs, units, name, bias=False, trainable = True, activation = None, reuse = False):
    return tf.layers.dense(inputs, units = units,
             kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
             activation = activation,
             trainable = trainable,
             use_bias = bias,
             reuse = reuse,
             name = name)

def res_block(maps, out_channel, name, block_stride=1):

    with tf.variable_scope(name) as scope:
        conv0 = tf.nn.relu(tf.layers.batch_normalization(conv2d(maps, out_channel, 'conv_0', kernel_size=3, strides =block_stride),training=False))
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(conv0, out_channel, 'conv_1', kernel_size=3, strides =1),training=False))

        maps = tf.layers.batch_normalization(conv2d(maps, out_channel, 'shortcut', kernel_size=1, strides =block_stride),training=False)

    return tf.nn.relu(conv1 + maps)

def model(input_tensor_batch, class_num):
    
    with tf.variable_scope('vgg16') as scope:

        conv1_0 = tf.nn.relu(conv2d(input_tensor_batch, 64, 'conv1_0', kernel_size=3, strides=1))
        conv1_1 = tf.nn.relu(conv2d(conv1_0, 64, 'conv1_1', kernel_size=3, strides=1))
        pool0 = tf.nn.max_pool(conv1_1, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv2_0 = tf.nn.relu(conv2d(pool0, 128, 'conv2_0', kernel_size=3, strides=1))
        conv2_1 = tf.nn.relu(conv2d(conv2_0, 128, 'conv2_1', kernel_size=3, strides=1))
        pool1 = tf.nn.max_pool(conv2_1, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv3_0 = tf.nn.relu(conv2d(pool1, 256, 'conv3_0', kernel_size=3, strides=1))
        conv3_1 = tf.nn.relu(conv2d(conv3_0, 256, 'conv3_1', kernel_size=3, strides=1))
        conv3_2 = tf.nn.relu(conv2d(conv3_1, 256, 'conv3_2', kernel_size=3, strides=1))
        pool2 = tf.nn.max_pool(conv3_2, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv4_0 = tf.nn.relu(conv2d(pool2, 512, 'conv4_0', kernel_size=3, strides=1))
        conv4_1 = tf.nn.relu(conv2d(conv4_0, 512, 'conv4_1', kernel_size=3, strides=1))
        conv4_2 = tf.nn.relu(conv2d(conv4_1, 512, 'conv4_2', kernel_size=3, strides=1))
        pool3 = tf.nn.max_pool(conv4_2, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv5_0 = tf.nn.relu(conv2d(pool3, 512, 'conv5_0', kernel_size=3, strides=1))
        conv5_1 = tf.nn.relu(conv2d(conv5_0, 512, 'conv5_1', kernel_size=3, strides=1))
        conv5_2 = tf.nn.relu(conv2d(conv5_1, 512, 'conv5_2', kernel_size=3, strides=1))
        pool4 = tf.nn.max_pool(conv5_2, [1,2,2,1], [1,2,2,1], padding='SAME')

        fc0 = tf.reshape(pool4, [-1,1*1*512])
        fc1 = tf.nn.relu(fullyConnect(fc0, 4096, "fc1"))
        fc2 = tf.nn.relu(fullyConnect(fc1, 4096, "fc2"))

        output = fullyConnect(fc2, class_num, name='output')

    return output

