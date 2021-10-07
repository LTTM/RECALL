import tensorflow as tf


def upsampling(inputs, output_shape):
    return tf.compat.v1.image.resize_bilinear(inputs, size=tf.cast([output_shape[1], output_shape[2]], tf.int32))


def conv2d(x, kernel_size, num_o, stride, name, channel_axis=3, biased=False):
    """
    Conv2d without BN or relu.
    """
    num_x = x.shape[channel_axis].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, stride, stride, 1]
        o = tf.nn.conv2d(x, w, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, channel_axis=3, biased=False):
    """
    Dilated conv2d without BN or relu.
    """
    num_x = x.shape[channel_axis].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def relu(x, name):
    return tf.nn.relu(x, name=name)


def add(x_l, name):
    return tf.add_n(x_l, name=name)


def max_pool2d(x, kernel_size, stride, name):
    k = [1, kernel_size, kernel_size, 1]
    s = [1, stride, stride, 1]
    return tf.nn.max_pool(x, k, s, padding='SAME', name=name)


def batch_norm(x, name, is_training, activation_fn, trainable=False):
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances) frozen,
    # and to not update the values provided by the pre-trained model by setting is_training=False.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.
    # Set trainable = False to remove them from trainable_variables.
    with tf.variable_scope(name + '/BatchNorm') as scope:
        o = tf.contrib.layers.batch_norm(
            x,
            scale=True,
            activation_fn=activation_fn,
            is_training=is_training,
            trainable=trainable,
            scope=scope)
        return o


def ASPP(x, num_o, dilations, channel_axis=3):
    o = []
    for i, d in enumerate(dilations):
        o.append(dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i+1), channel_axis=channel_axis, biased=True))
    return add(o, name='aspp/add')