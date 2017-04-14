import numpy as np
import tensorflow as tf
import math

def next_batch(X, y, size):
    perm = np.random.permutation(X.shape[0])
    for i in np.arange(0, X.shape[0], size):
        yield (X[perm[i:i + size]], y[perm[i:i + size]])


def weight_variable(shape, name='W'):
    # He et. al initialization
    n = shape[0] * shape[1] * shape[2] # multiplying patch_x, patch_y and input_channel
    # Note: The paper suggets that we should have a different initialization for first layer
    # since the first layer does not have input which is the output of ReLU
    init_term = math.sqrt(2.0/n)
    initial = tf.truncated_normal(shape, stddev=init_term)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def loss_accuracy(cross_entropy_count, accuracy_count, x, y_, keep_prob, phase_train, X, Y, batch_size):
    c, l = 0, 0
    for batch_xs, batch_ys in next_batch(X, Y, batch_size):
        feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 1.0, phase_train: False}
        l += cross_entropy_count.eval(feed_dict=feed_dict)
        c += accuracy_count.eval(feed_dict=feed_dict)
    return float(l) / float(Y.shape[0]), float(c) / float(Y.shape[0])


def max_pool_layer(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

    
def conv_layer(input, channels_in, channels_out, phase_train, name='conv', patch_x=15, patch_y=15):
    with tf.name_scope(name):
        # 15x15 patch, 8 channels, 32 output channels
        w = weight_variable([patch_x, patch_y, channels_in, channels_out], name='W')
        b = bias_variable([channels_out], name='B')
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b
        bn = batch_norm(conv, channels_out, phase_train, name + '_bn')
        act = tf.nn.relu(bn)
#         tf.summary.histogram('weights', w)
#         tf.summary.histogram('biases', b)
#         tf.summary.histogram('activations', act)
        return act
    
def fc_layer(input, channels_in, channels_out, phase_train, name='fc'):
    with tf.name_scope(name):
        w = weight_variable([channels_in, channels_out], name='W')
        b = bias_variable([channels_out], name='b')
        fc = tf.matmul(input, w) + b
#        bn = batch_norm(fc, channels_out, phase_train, name + '_bn')
#         tf.summary.histogram('weights', w)
#         tf.summary.histogram('biases', b)
        return fc # return tf.nn.relu(bn)

def batch_norm(x, n_out, phase_train, name='bn'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.6)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-2)
    return normed
