import tensorflow as tf

weight_no = 1
bias_no = 1
def new_weights(shape):
    global weight_no
    w_name = "w" + `weight_no`
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05), w_name)
    tf.summary.histogram(w_name, weights)
    weight_no += 1
    return weights

def new_biases(length):
    global bias_no
    b_name = "b" + `bias_no`
    bias = tf.Variable(tf.constant(0.05, shape=[length]), b_name)
    tf.summary.histogram(b_name, bias)
    bias_no += 1
    return bias
