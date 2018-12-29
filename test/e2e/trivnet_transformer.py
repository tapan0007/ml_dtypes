
from trivnet_common import *


### Parameters #########################################################

# From transformer offical model
LAYER_NORM_EPS = 1e-6

# HIDDEN_SIZE = 32 showed less error for multiple dimension configurations with
# input ranges of tfloat16-wmin-0.01-wmax0.01-imin-0.1-imax0.1.
HIDDEN_SIZE = 32
#HIDDEN_SIZE = conf.NUMHID   

NUM_HIDDEN_LAYERS = 1

batch_input_len = conf.BATCHSIZE * conf.INPUTLEN
batch_output_len = conf.BATCHSIZE * conf.OUTPUTLEN

########################################################################



np.random.seed(15213)

# input: [W, C]
# output: [W, C]
def layer_normalization(inputs):

    layer_norm_scale = conf.gen_array_rand(conf.WMIN, conf.WMAX, shape = inputs.shape)
    layer_norm_bias = conf.gen_array_rand(conf.WMIN, conf.WMAX, shape = inputs.shape)
 
    num_channels = inputs.shape[-1]

    def broadcast_c(v, _name):
        #broadcast_w = np.full([1,1,1,num_channels],1.0)
        #return tf.nn.conv2d(v, broadcast_w, strides=[1,1,1,1], padding='SAME',name='broadcast_' + _name)
        broadcast_w = np.full([1, num_channels], 1.0).astype(conf.npDataType)
        return tf.matmul(v, broadcast_w, name='broadcast_' + _name)


    inputs_sum = tf.reduce_sum(inputs, axis=[-1], keepdims=True, name='sum')

    factor = 1.0 / HIDDEN_SIZE
    mean = tf.multiply(inputs_sum, factor, name='mean')

    mean_b = broadcast_c(mean, "b1")

    residuals = tf.subtract(inputs, mean_b, name='residuals')

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

    return pre_outputs



# Assum conf.INPUTLEN == conf.OUTPUTLEN
# input: [conf.BATCHSIZE * conf.INPUTLEN, conf.NUMHID]
# alpha = pick a value
# filter_kernel: [conf.NUMHID, alpha]
# filter_bias: [alpha]
# output_kernel: [alpha, conf.NUMHID]
# output_bias: [conf.NUMHID]
# output: [conf.BATCHSIZE * conf.INPUTLEN, conf.NUMHID]

def feed_forward_network(inputs, layer_name_prefix):
    
    alpha = conf.NUMHID
    filter_kernel_shape = [conf.NUMHID, alpha]
    filter_bias_shape = [alpha]
    output_kernel_shape = [alpha, conf.NUMHID]
    output_bias_shape = [conf.NUMHID]
    
    filter_kernel_val = conf.gen_array_linspace(conf.WMIN, conf.WMAX, filter_kernel_shape)
    filter_kernel = conf.gen_variable_tensor(name = '%s_filter_kernel' % layer_name_prefix, initializer = filter_kernel_val)

    filter_bias_val = conf.gen_array_linspace(conf.WMIN, conf.WMAX, filter_bias_shape)
    filter_bias = conf.gen_variable_tensor(name = '%s_filter_bias' % layer_name_prefix, initializer = filter_bias_val)

    output_kernel_val = conf.gen_array_linspace(conf.WMIN, conf.WMAX, output_kernel_shape)
    output_kernel = conf.gen_variable_tensor(name = '%s_output_kernel' % layer_name_prefix, initializer = output_kernel_val)

    output_bias_val = conf.gen_array_linspace(conf.WMIN, conf.WMAX, output_bias_shape)
    output_bias = conf.gen_variable_tensor(name = '%s_output_bias' % layer_name_prefix, initializer = output_bias_val)    

    with tf.name_scope('filter_layer'):
        filter_layer = tf.nn.bias_add(tf.matmul(inputs, filter_kernel, name='matmul'),
            filter_bias, name='bias_add')
        filter_layer = tf.nn.relu(filter_layer)
    with tf.name_scope('output_layer'):
        output = tf.nn.bias_add(tf.matmul(filter_layer, output_kernel, name='matmul'),
            output_bias, name = 'bias_add')

    return output



def multihead_attention(b_input_x_r, b_input_y_r, b_bias_br_arr):
    
    # constants
    q_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    k_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    tr_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])    
    q_kernel = q_kernel / np.sqrt(conf.HEADSIZE)    
    v_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID]) 
    ones_for_br = np.ones([1, conf.OUTPUTLEN]).astype(conf.npDataType)

    # graph
    b_q_heads_tiled = tf.matmul(b_input_x_r, q_kernel,
        name='b_q_heads_tiled')
    b_k_heads_tiled = tf.matmul(b_input_y_r, k_kernel,
        name='b_k_heads_tiled')
    b_v_t_heads_tiled = tf.matmul(v_kernel, b_input_y_r, transpose_b=True,
        name='b_v_heads_t_tiled')

    # no way to make this efficient but matrices are small
    b_weighted_v_t_list = []
    for i, (bst_in, bst_out) in enumerate(
            zip(range(0, batch_input_len, conf.INPUTLEN),
                range(0, batch_output_len, conf.OUTPUTLEN))):
        with tf.name_scope('batch_%d' % i):
            weighted_v_heads_list = []
            for j, hst in enumerate(range(0, conf.NUMHID, conf.HEADSIZE)):
                with tf.name_scope('head_%d' % j):
                    # ME cannot handle channel_slice(w_slice(x)). w_slice(channel_slice(x)) is fine.
                    #q_head_batch = tf.slice(b_q_heads_tiled, [bst_in, 0], [conf.INPUTLEN, -1], name='q_head_batch')
                    #q_head = q_head_batch[:, hst:hst+conf.HEADSIZE]
                    q_head_c_slice = b_q_heads_tiled[:, hst:hst+conf.HEADSIZE]
                    q_head = tf.slice(q_head_c_slice, [bst_in, 0], [conf.INPUTLEN, -1], name='q_head')
                                                                           
                    #k_head_batch = tf.slice(b_k_heads_tiled, [bst_out, 0], [conf.OUTPUTLEN, -1], name='k_head_batch')
                    #k_head = k_head_batch[:, hst:hst+conf.HEADSIZE]
                    #k_head_c_slice = b_k_heads_tiled[:, hst:hst+conf.HEADSIZE]
                    #wrapped by tf.add to avoid ME bug. ME totally loose this value.
                    k_head_c_slice = tf.add(b_k_heads_tiled[:, hst:hst+conf.HEADSIZE], 0.0)
                    k_head = tf.slice(k_head_c_slice, [bst_out, 0], [conf.OUTPUTLEN, -1], name='k_head_batch')
                    
                    qk_head = tf.matmul(q_head, k_head, transpose_b=True, name='qk_head') # (16, 16)

                    # add bias
                    # skipping overflow prevention for now because that will need tf.reduce_max
                    bias_br = b_bias_br_arr[i] # (16, 16)
                    qk_bias_head = tf.add(qk_head, bias_br)
                    qk_exp_head = tf.exp(qk_bias_head) # (16, 16)
                    norm_factor = tf.reduce_sum(qk_exp_head, axis=1, keepdims=True) # (16, 1)
                    norm_factor_rec = tf.reciprocal(norm_factor) # (16, 1)
                    norm_factor_rec_br = tf.matmul(norm_factor_rec, ones_for_br, name='norm_factor_rec_br') # (16, 16)
                    weight_head = tf.multiply(qk_exp_head, norm_factor_rec_br) # (16, 16)

                    #v_t_head_batch = tf.slice(b_v_t_heads_tiled, [hst, 0], [conf.HEADSIZE, -1], name='v_t_head_batch')
                    #v_t_head = v_t_head_batch[:, bst_out:bst_out+conf.OUTPUTLEN] # (16, 64)
                    v_t_head_c_slice = b_v_t_heads_tiled[:, bst_out:bst_out+conf.OUTPUTLEN] # (16, 64)
                    v_t_head = tf.slice(v_t_head_c_slice, [hst, 0], [conf.HEADSIZE, -1], name='v_t_head_batch')
                                                
                    weighted_v_head = tf.matmul(weight_head, v_t_head, transpose_b=True, name='weighted_v_head') # (16, 64)
                    weighted_v_heads_list.append(weighted_v_head)
            # merge heads
            weighted_v = tf.concat(weighted_v_heads_list, axis=1) # (16, 1024)
            weighted_v_t = tf.transpose(weighted_v, perm=[1, 0]) # (1024, 16)
            b_weighted_v_t_list.append(weighted_v_t)
    b_weighted_v_t_tiled = tf.concat(b_weighted_v_t_list, axis=1, # (1024, 64)
        name='b_weighted_v_t_tiled')
    b_weighted_v_tiled = tf.transpose(b_weighted_v_t_tiled, perm=[1, 0], # (64, 1024)
        name='b_weighted_v_tiled')

    output = tf.matmul(b_weighted_v_tiled, tr_kernel,
        name='mhatt_output') # (64, 1024)

    return output


# Avoid too long scope names since the runtime has a limitation of file name length.
def encoder_block(inputs, attention_bias, layer_name_prefix):
    with tf.name_scope('self_att'):
        #layer_name = 'encoder_stack/layer_%d/self_attention' % idx
        with tf.name_scope('lm'):
            inputs_norm = layer_normalization(inputs)
        #layer_name += '/self_attention'
        with tf.name_scope('mhatt'):
            self_att = multihead_attention(inputs_norm, inputs_norm,
                attention_bias)
            inputs = tf.add(self_att, inputs, name='resadd')

    #layer_name = 'encoder_stack/layer_%d/ffn' % idx
    with tf.name_scope('ffn_blk'):
        with tf.name_scope('lm'):
            inputs_norm = layer_normalization(inputs)
        with tf.name_scope('ffn'):
            outputs = feed_forward_network(inputs_norm, layer_name_prefix)
            outputs = tf.add(outputs, inputs, name='resadd')
        
    return outputs
    

def encoder_stack(inputs, attention_bias):
    # encoder blocks
    for idx in range(NUM_HIDDEN_LAYERS):
        #print('l%d' % idx)
        with tf.name_scope('l%d' % idx):
            inputs = encoder_block(inputs, attention_bias, 'el'+str(idx))

    # layer normalization after encoding layers
    with tf.name_scope('lm'):
        encoder_outputs = layer_normalization(inputs)

    return encoder_outputs
    
    
def decoder_block(inputs, encoder_outputs, encoder_attention_bias, decoder_attention_bias, layer_name_prefix):
    with tf.name_scope('self_att'):
        #layer_name = 'decoder_stack/layer_%d/self_attention' % idx
        with tf.name_scope('lm'):
            inputs_norm = layer_normalization(inputs)
        #layer_name += '/self_attention'
        with tf.name_scope('mhatt'):
            self_att = multihead_attention(inputs_norm, inputs_norm,
                decoder_attention_bias) # Do we need bias_add_func=tf.add ?
            inputs = tf.add(self_att, inputs, name='resadd')

    with tf.name_scope('encdec_att'):
        #layer_name = 'decoder_stack/layer_%d/encdec_attention' % idx
        with tf.name_scope('lm'):
            inputs_norm = layer_normalization(inputs)
        #layer_name += '/attention'
        with tf.name_scope('att'):
            att = multihead_attention(inputs_norm, encoder_outputs,
                encoder_attention_bias)
            inputs = tf.add(att, inputs, name='resadd')

    with tf.name_scope('ffn_blk'):
        #layer_name = 'decoder_stack/layer_%d/ffn' % idx
        with tf.name_scope('lm'):
            inputs_norm = layer_normalization(inputs)
        outputs = feed_forward_network(inputs_norm, layer_name_prefix)
        outputs = tf.add(outputs, inputs, name='resadd')
        
    return outputs    
    
   
# layer_normalization has a high degree of value errors if it is run alone
# although the number distribution looks okay.
#
# "2-layer_norm": [
# "trivnet_transformer",
# "tfloat16-wmin-0.01-wmax0.01-imin-0.1-imax0.1-batchsize4-inputlen16-outputlen16-headsize4-numhid16-neginf-100000000-test_layer_norm",  "layer_norm" ,
# "--partition none "
# "--executors wave all --images linspace:0-0.1 {} --wavegraph_checks structure data-race".format(MEv2("generic-noverify")),
# " --diff_options '--tolerance 3.0 1e-5' "
# "--check_against_ref all_available --input_files input:0=trivnet_input:0.npy "
# ],  

def test_layer_normalization():

    # layer_normalization test 

    #W = 1
    #C = 1024 #1024
    W = conf.BATCHSIZE * conf.INPUTLEN
    C = conf.NUMHID
    input_shape = [W,C] 

    i0 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
    #i1 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input_const")

    with tf.name_scope('%s' % conf.netName):
        output = layer_normalization(i0)
       
        
    i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)
    conf.gen_graph(output, {"input:0" : i0val}, need_freezing=False)

    
def test_multihead_attention():
    
    # example
    # input_y_r: 128 x 512
    # input_x_r: 64 x 512
    # input_bias_br_n : 16 x 32
    # v_kernel_t : 512 x 512
    # output: 512 x 64


    '''
    v_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    tr_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    np_v_kernel_t = v_kernel.T
    '''

    b_bias_np = np.zeros([conf.BATCHSIZE, conf.OUTPUTLEN]).astype(conf.npDataType)
    b_bias_br_np = b_bias_np[:, :, np.newaxis] \
        .dot(np.ones([conf.INPUTLEN, 1]).astype(conf.npDataType).T) \
        .transpose([0, 2, 1])
        
    b_input_x_np = conf.gen_array_rand(conf.IMIN, conf.IMAX,    shape=[conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID])
    b_input_x_r_np = b_input_x_np.reshape([batch_input_len, conf.NUMHID])
    #b_input_y_r_np = b_input_y_np.reshape([batch_input_len, conf.NUMHID])   
    #b_input_y_r_np = b_input_y_np.reshape([batch_output_len, conf.NUMHID])    
    b_input_y_r_np = b_input_x_r_np
        
    b_input_x_r = tf.placeholder(conf.tfDataType, name='input_x_r',
        shape=[batch_input_len, conf.NUMHID])
    b_input_y_r = tf.placeholder(conf.tfDataType, name='input_y_r', shape=[batch_output_len, conf.NUMHID])

    b_bias_br = [
        tf.placeholder(conf.tfDataType, name='input_bias_br_%d'%i,
            shape=[conf.INPUTLEN, conf.OUTPUTLEN])
        for i in range(conf.BATCHSIZE)
    ]

    with tf.name_scope('%s' % conf.netName):
        mha_out = multihead_attention(b_input_x_r, b_input_y_r, b_bias_br)
        
    output = tf.identity(mha_out, name = conf.netName + "/output")
        

    input_data = {
        'input_x_r:0': b_input_x_r_np,
        'input_y_r:0': b_input_x_r_np
        }
    for i in range(conf.BATCHSIZE):
        input_data['input_bias_br_%d:0'%i] = b_bias_br_np[i]

    conf.gen_graph(output, input_data, need_freezing=False)
    

# Encoder layer test
def encoder_test(testName):
    
    enc_input = tf.placeholder(conf.tfDataType, name='input', shape=[batch_input_len, conf.NUMHID])
    b_bias_br = [
        tf.placeholder(conf.tfDataType, name='input_bias_br_%d'%i,
            shape=[conf.INPUTLEN, conf.OUTPUTLEN])
        for i in range(conf.BATCHSIZE)
    ]

    with tf.name_scope('%s' % conf.netName):
        if testName == 'encoder_layer':
            enc_out = encoder_block(enc_input, b_bias_br, 0)
        else: 
            enc_out = encoder_stack(enc_input, b_bias_br)
        
    output = tf.identity(enc_out, name = conf.netName + "/output")
        
    b_input_x_np = conf.gen_array_rand(conf.IMIN, conf.IMAX,    shape=[conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID])
    b_input_x_r_np = b_input_x_np.reshape([batch_input_len, conf.NUMHID])
    b_bias_np = np.zeros([conf.BATCHSIZE, conf.OUTPUTLEN]).astype(conf.npDataType)
    b_bias_np[0, 5] = conf.NEGINF
    b_bias_np[1, 8] = conf.NEGINF
    b_bias_np[2, 9] = conf.NEGINF
    b_bias_np[3, 2] = conf.NEGINF
    b_bias_br_np = b_bias_np[:, :, np.newaxis] \
        .dot(np.ones([conf.INPUTLEN, 1]).astype(conf.npDataType).T) \
        .transpose([0, 2, 1])

    input_data = {
        'input:0': b_input_x_r_np
        }
    for i in range(conf.BATCHSIZE):
        input_data['input_bias_br_%d:0'%i] = b_bias_br_np[i]

    conf.gen_graph(output, input_data, need_freezing=True)
    
    
    
def decoder_test(testName):

    dec_input = tf.placeholder(conf.tfDataType, name='input', shape=[batch_input_len, conf.NUMHID])
    enc_output = tf.placeholder(conf.tfDataType, name='input_from_enc', shape=[batch_input_len, conf.NUMHID])
    enc_bias_br = [
        tf.placeholder(conf.tfDataType, name='input_enc_bias_br_%d'%i,
            shape=[conf.INPUTLEN, conf.OUTPUTLEN])
        for i in range(conf.BATCHSIZE)
    ]
    dec_bias_br = [
        tf.placeholder(conf.tfDataType, name='input_dec_bias_br_%d'%i,
            shape=[conf.INPUTLEN, conf.OUTPUTLEN])
        for i in range(conf.BATCHSIZE)
    ]

    with tf.name_scope('%s' % conf.netName):
        if testName == 'decoder_layer':
            dec_out = decoder_block(dec_input, enc_output, enc_bias_br, dec_bias_br, 0)
        else:
            pass 
            #dec_out = encoder_stack(enc_input, b_bias_br)
        
    output = tf.identity(dec_out, name = conf.netName + "/output")
        
    b_input_x_np = conf.gen_array_rand(conf.IMIN, conf.IMAX,    shape=[conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID])
    b_input_x_r_np = b_input_x_np.reshape([batch_input_len, conf.NUMHID])
    b_bias_np = np.zeros([conf.BATCHSIZE, conf.OUTPUTLEN]).astype(conf.npDataType)
    b_bias_np[0, 5] = conf.NEGINF
    b_bias_np[1, 8] = conf.NEGINF
    b_bias_np[2, 9] = conf.NEGINF
    b_bias_np[3, 2] = conf.NEGINF
    b_bias_br_np = b_bias_np[:, :, np.newaxis] \
        .dot(np.ones([conf.INPUTLEN, 1]).astype(conf.npDataType).T) \
        .transpose([0, 2, 1])

    input_data = {
        'input:0': b_input_x_r_np,
        'input_from_enc:0': b_input_x_r_np
        }
    for i in range(conf.BATCHSIZE):
        input_data['input_enc_bias_br_%d:0'%i] = b_bias_br_np[i]
    for i in range(conf.BATCHSIZE):
        input_data['input_dec_bias_br_%d:0'%i] = b_bias_br_np[i]        

    conf.gen_graph(output, input_data, need_freezing=True)      
        
    
testName = conf.testName.lower()
print('TEST_NAME: ' + testName)


if testName == 'encoder_layer' or testName == 'encoder_stack':
    encoder_test(testName)
    
elif testName == 'decoder_layer':
    decoder_test(testName)

elif testName == 'layer_norm':
    test_layer_normalization()        

elif testName == 'mhatt':
    test_multihead_attention()
    
else:    
    assert(False and 'No test specified')

