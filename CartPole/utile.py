import tensorflow as tf


def linear(input, output_dim, initializer, activation_fn, name):
    with tf.variable_scope(name):
        input_dim = input.shape[1]
        w = tf.get_variable('w', [input_dim, output_dim], tf.float32, initializer)
        out = tf.matmul(input, w)
        b = tf.get_variable('b', [output_dim], tf.float32, tf.constant_initializer(0.0))
        out =  tf.nn.bias_add(out, b)
        if activation_fn:
            out = activation_fn(out)
        return out, w, b


