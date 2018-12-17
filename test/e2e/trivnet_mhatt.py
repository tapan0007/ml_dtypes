# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - attention block

from trivnet_common import *


def multihead_attention_reference(x, y, bias,
    q_kernel, k_kernel, v_kernel, tr_kernel):
    q_kernel = q_kernel / np.sqrt(conf.HEADSIZE)
    start_range = range(0, len(q_kernel), conf.HEADSIZE)

    '''
    safest method at this point (slow?):
    split in numpy then operate on each head individually
    '''
    q_kernel_heads = [q_kernel[:, st:st+conf.HEADSIZE] for st in start_range]
    k_kernel_heads = [k_kernel[:, st:st+conf.HEADSIZE] for st in start_range]
    v_kernel_heads = [v_kernel[:, st:st+conf.HEADSIZE] for st in start_range]
    q_heads = [tf.matmul(x, qk, name='q_head_%d'%i)
        for i, qk in enumerate(q_kernel_heads)]
    k_heads = [tf.matmul(y, kk, name='k_head_%d'%i)
        for i, kk in enumerate(k_kernel_heads)]
    v_heads = [tf.matmul(y, vk, name='v_head_%d'%i)
        for i, vk in enumerate(v_kernel_heads)]

    dot_heads = [tf.matmul(q, k, transpose_b=True, name='dot_head_%d'%i)
        for i, (q, k) in enumerate(zip(q_heads, k_heads))]

    # mask
    wt_heads = [tf.nn.softmax(tf.nn.bias_add(d, bias), name='w_head_%d'%i)
        for i, d in enumerate(dot_heads)]

    # merge heads
    wv_heads = [tf.matmul(w, v, name='wv_head_%d'%i)
        for i, (w, v) in enumerate(zip(wt_heads, v_heads))]
    concat_heads = tf.concat(wv_heads, axis=-1, name='concat_heads')
    outputs = tf.matmul(concat_heads, tr_kernel, name='outputs')
    return outputs

if __name__ == '__main__':
    # reference
    np.random.seed(15213)
    q_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    k_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    v_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])
    tr_kernel = conf.gen_array_rand(conf.WMIN, conf.WMAX,
        shape=[conf.NUMHID, conf.NUMHID])

    b_input_x_np = conf.gen_array_rand(conf.IMIN, conf.IMAX,
        shape=[conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID])
    b_input_y_np = conf.gen_array_rand(conf.IMIN, conf.IMAX,
        shape=[conf.BATCHSIZE, conf.OUTPUTLEN, conf.NUMHID])
    b_bias_np = np.zeros([conf.BATCHSIZE, conf.OUTPUTLEN]).astype(conf.npDataType)
    b_bias_np[0, 5] = conf.NEGINF
    b_bias_np[1, 8] = conf.NEGINF
    b_bias_np[2, 9] = conf.NEGINF
    b_bias_np[3, 2] = conf.NEGINF

    with tf.Graph().as_default():
        input_x = tf.placeholder(conf.tfDataType, [conf.INPUTLEN, conf.NUMHID])
        input_y = tf.placeholder(conf.tfDataType, [conf.OUTPUTLEN, conf.NUMHID])
        bias = tf.placeholder(conf.tfDataType, [conf.OUTPUTLEN])
        att_out_ref = multihead_attention_reference(input_x, input_y, bias,
            q_kernel, k_kernel, v_kernel, tr_kernel)
        sess = tf.Session()
        b_att_ref_np = []
        for i in range(conf.BATCHSIZE):
            att_ref_np = sess.run(att_out_ref, feed_dict={
                input_x: b_input_x_np[i],
                input_y: b_input_y_np[i],
                bias: b_bias_np[i]})
            b_att_ref_np.append(att_ref_np)
        b_att_ref_np = np.asarray(b_att_ref_np)
        print(b_att_ref_np, b_att_ref_np.shape)

    print('reimplementation:')
    batch_input_len = conf.BATCHSIZE * conf.INPUTLEN
    batch_output_len = conf.BATCHSIZE * conf.OUTPUTLEN
    ones_for_br = np.ones([1, conf.OUTPUTLEN]).astype(conf.npDataType)
    with tf.Graph().as_default():
        q_kernel = q_kernel / np.sqrt(conf.HEADSIZE)
        b_bias_br_np = b_bias_np[:, :, np.newaxis] \
            .dot(np.ones([conf.INPUTLEN, 1]).astype(conf.npDataType).T) \
            .transpose([0, 2, 1])
        b_input_x_r_np = b_input_x_np.reshape([batch_input_len, conf.NUMHID])
        b_input_y_r_np = b_input_y_np.reshape([batch_output_len, conf.NUMHID])

        np_v_kernel_t = v_kernel.T

        v_kernel_t = tf.placeholder(conf.tfDataType, name='v_kernel_t', shape=np_v_kernel_t.shape)

        b_input_x_r = tf.placeholder(conf.tfDataType, name='input_x_r',
            shape=[batch_input_len, conf.NUMHID])
        b_input_y_r = tf.placeholder(conf.tfDataType, name='input_y_r',
            shape=[batch_output_len, conf.NUMHID])
        b_bias_br = [
            tf.placeholder(conf.tfDataType, name='input_bias_br_%d'%i,
                shape=[conf.INPUTLEN, conf.OUTPUTLEN])
            for i in range(conf.BATCHSIZE)
        ]

        b_q_heads_tiled = tf.matmul(b_input_x_r, q_kernel,
            name='%s/part1/b_q_heads_tiled' % conf.netName)
        b_k_heads_tiled = tf.matmul(b_input_y_r, k_kernel,
            name='%s/part1/b_k_heads_tiled' % conf.netName)
        b_v_t_heads_tiled = tf.matmul(v_kernel_t, b_input_y_r, transpose_b=True,
            name='%s/part1/b_v_heads_t_tiled' % conf.netName)

        with tf.name_scope('%s/part2' % conf.netName):
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
                            bias_br = b_bias_br[i] # (16, 16)
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
            name='%s/part3/output' % conf.netName) # (64, 1024)

        input_data = {
            'input_x_r:0': b_input_x_r_np,
            'input_y_r:0': b_input_y_r_np,
            'v_kernel_t:0': np_v_kernel_t,
            }
        for i in range(conf.BATCHSIZE):
            input_data['input_bias_br_%d:0'%i] = b_bias_br_np[i]

        sess = tf.Session()
        b_att_t_np = sess.run(output, feed_dict=input_data)
        b_att_np = b_att_t_np.reshape(conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID)
        print(b_att_np, b_att_np.shape)
        error = b_att_ref_np.ravel() - b_att_np.ravel()
        print('rms error:', np.sqrt((error * error).mean()))

        conf.gen_graph(output, input_data, need_freezing=False)
