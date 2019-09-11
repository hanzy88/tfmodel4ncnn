
import numpy as np
from hyper_parameters import *

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
        conv0 = tf.nn.relu(tf.layers.batch_normalization(conv2d(maps, out_channel, 'conv_0', kernel_size=3, strides =block_stride),training=True))
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(conv0, out_channel, 'conv_1', kernel_size=3, strides =1),training=True))

        maps = tf.layers.batch_normalization(conv2d(maps, out_channel, 'shortcut', kernel_size=1, strides =block_stride),training=True)

    return tf.nn.relu(conv1 + maps)

def inference(input_tensor_batch, n, reuse):
    
    with tf.variable_scope('resnet32',reuse = reuse) as scope:

        conv0 = tf.nn.relu(tf.layers.batch_normalization(conv2d(input_tensor_batch, 16, 'conv0', kernel_size=3, strides=1),training=True))

        #pool0 = tf.nn.max_pool(conv0, [1,3,3,1], [1,2,2,1], padding='SAME')

        resnet1_0 = res_block(conv0, 16, 'resnet1_0')
        resnet1_1 = res_block(resnet1_0, 16, 'resnet1_1')
        resnet1_2 = res_block(resnet1_1, 16, 'resnet1_2')
        resnet1_3 = res_block(resnet1_2, 16, 'resnet1_3')
        resnet1_4 = res_block(resnet1_3, 16, 'resnet1_4')

        resnet2_0 = res_block(resnet1_4, 32, 'resnet2_0', block_stride = 2)
        resnet2_1 = res_block(resnet2_0, 32, 'resnet2_1')
        resnet2_2 = res_block(resnet2_1, 32, 'resnet2_2')
        resnet2_3 = res_block(resnet2_2, 32, 'resnet2_3')
        resnet2_4 = res_block(resnet2_3, 32, 'resnet2_4')

        resnet3_0 = res_block(resnet2_4, 64, 'resnet3_0', block_stride = 2)
        resnet3_1 = res_block(resnet3_0, 64, 'resnet3_1')
        resnet3_2 = res_block(resnet3_1, 64, 'resnet3_2')
        resnet3_3 = res_block(resnet3_2, 64, 'resnet3_3')
        resnet3_4 = res_block(resnet3_3, 64, 'resnet3_4')

        pool1 = tf.reduce_mean(resnet3_4, reduction_indices=[1,2], name='global_avg_pool')
        output = fullyConnect(pool1, 10, name='output')

    return output


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
