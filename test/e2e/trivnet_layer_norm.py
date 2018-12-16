
from trivnet_common import *


# N, H, W, C
C = 1024
HIDDEN_SIZE = C
input_shape = [1,1,1,C] 
layer_norm_scale_shape = [1,1,1,C]
layer_norm_bias_shape = [1,1,1,C]
broad_cast_weights = np.full([1,1,1,input_shape[3]],1.0)

# from transformer official model
LAYER_NORM_EPS = 1e-6

def broadcast_c(v, _name):
	broadcast_w = broad_cast_weights
	return tf.nn.conv2d(v, broadcast_w, strides=[1,1,1,1], padding='SAME',name='broadcast_' + _name)


i0 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input_const")

with tf.name_scope(conf.netName):

	inputs = i0

	# layer_norm_scale is originally a constant tensor but compiler doesn't support
	# it yet. Used an input instead.
	layer_norm_scale = i1

	# layer_norm_bias is originally a constant tensor but compiler doesn't support
	# it yet. Used an input instead.
	layer_norm_bias = i1

	inputs_sum = tf.reduce_sum(inputs, axis=[-1], keepdims=True, name='sum')
	
	factor = 1.0 / HIDDEN_SIZE
	mean = tf.multiply(inputs_sum, factor, name='mean')

	mean_b = broadcast_c(mean, "b1")

	#residuals = tf.subtract(inputs, mean_b, name='residuals')
	residuals = tf.add(inputs, mean_b, name='residuals')

	# ME cannot multiply same fmaps
	res_x_res = tf.multiply(residuals, residuals, name = "res_x_res")
	
	residuals_squared_sum = tf.reduce_sum(res_x_res, axis=[-1],
		keepdims=True, name='residuals_squared_sum')
		
	var = tf.multiply(residuals_squared_sum, factor, name='mult_var')
	
	rsqrt_ret = tf.rsqrt(var + LAYER_NORM_EPS, name="rsqrt_ret")
	
	norm_inputs = tf.multiply(
		residuals, 
		broadcast_c(rsqrt_ret , "b2"),
		'normalized')
		
	norm_scale = tf.multiply(norm_inputs, layer_norm_scale, name = 'norm_scale')
	pre_outputs = tf.add(norm_scale, layer_norm_bias, "pre_output")
	
	output = tf.identity(pre_outputs, "output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)
i1val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, {"input:0" : i0val, "input_const:0" : i1val}, need_freezing=False)
