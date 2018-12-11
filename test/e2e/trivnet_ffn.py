
from trivnet_common import *

# Node: Enable 1024, 4096. It has qemu failure.
ILEN = 512 #1024 
FLEN = 2048 #4096 
input_shape = [1,ILEN]
filter_kernel_shape = [ILEN, FLEN]
filter_bias_shape = [FLEN]
output_kernel_shape = [FLEN, ILEN]
output_bias_shape = [ILEN]

WMIN = 0.0
WMAX = 0.01

i0 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")

with tf.name_scope(conf.netName):
    
    inputs = i0
    
    filter_kernel_val = conf.gen_array_linspace(WMIN, WMAX, filter_kernel_shape)
    filter_kernel = conf.gen_variable_tensor(name = conf.netName + "/filter_kernel", initializer = filter_kernel_val)
    
    filter_bias_val = conf.gen_array_linspace(WMIN, WMAX, filter_bias_shape)
    filter_bias = conf.gen_variable_tensor(name = conf.netName + "/filter_bias", initializer = filter_bias_val)

    output_kernel_val = conf.gen_array_linspace(WMIN, WMAX, output_kernel_shape)
    output_kernel = conf.gen_variable_tensor(name = conf.netName + "/output_kernel", initializer = output_kernel_val)

    output_bias_val = conf.gen_array_linspace(WMIN, WMAX, output_bias_shape)
    output_bias = conf.gen_variable_tensor(name = conf.netName + "/output_bias", initializer = output_bias_val)

    with tf.name_scope('filter_layer'):
        filter_layer = tf.nn.bias_add(tf.matmul(inputs, filter_kernel, name='matmul'),
            filter_bias, name='bias_add')
        filter_layer = tf.nn.relu(filter_layer)
    with tf.name_scope('output_layer'):
        pre_output = tf.nn.bias_add(tf.matmul(filter_layer, output_kernel, name='matmul'),
            output_bias, name = 'bias_add')

    output = tf.identity(pre_output, "output")


i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, {"input:0" : i0val}, need_freezing=True)



