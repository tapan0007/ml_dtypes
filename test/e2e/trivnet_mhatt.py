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

        q_kernel_t = q_kernel.T
        k_kernel_t = k_kernel.T
        v_kernel_t = v_kernel.T
        tr_kernel_t = tr_kernel.T

        b_input_x_r = tf.placeholder(conf.tfDataType, name='input_x_r',
            shape=[batch_input_len, conf.NUMHID])
        b_input_y_r = tf.placeholder(conf.tfDataType, name='input_y_r',
            shape=[batch_output_len, conf.NUMHID])
        b_bias_br = tf.placeholder(conf.tfDataType, name='input_bias_br',
            shape=[conf.BATCHSIZE, conf.INPUTLEN, conf.OUTPUTLEN])

        # (1024, 1024) * (1024, 64) --> (1024, 64)
        # scheduling:
        # (1024, 1024) * (1024, 64)
        #
        # [(512, 1024) * (1024, 64),
        #  (512, 1024) * (1024, 64)]
        #
        # each one: (512, 128) * (128, 64) + (512, 128) * (128, 64) + ... (8 of them)
        b_q_t_heads_tiled = tf.matmul(q_kernel_t, b_input_x_r, transpose_b=True, # (1024, 64), 88.88%
            name='%s/part1/b_q_heads_t_tiled' % conf.netName)
        b_k_t_heads_tiled = tf.matmul(k_kernel_t, b_input_y_r, transpose_b=True, # (1024, 64), 88.88%
            name='%s/part1/b_k_heads_t_tiled' % conf.netName)
        b_v_t_heads_tiled = tf.matmul(v_kernel_t, b_input_y_r, transpose_b=True, # (1024, 64), 88.88%
            name='%s/part1/b_v_heads_t_tiled' % conf.netName)

        # must have transpose for efficiency reasons
        b_q_heads_tiled = tf.transpose(b_q_t_heads_tiled, perm=[1, 0], # (64, 1024)
            name='%s/part1/b_q_heads_tiled' % conf.netName)
        b_k_heads_tiled = tf.transpose(b_k_t_heads_tiled, perm=[1, 0], # (64, 1024)
            name='%s/part1/b_k_heads_tiled' % conf.netName)
        b_v_t_heads_tiled = tf.identity(b_v_t_heads_tiled,
            name='%s/part1/b_v_heads_t_tiled_for_partition' % conf.netName)
        b_bias_br = tf.identity(b_bias_br,
            name='%s/part1/b_bias_br_for_partition' % conf.netName)

        with tf.name_scope('%s/part2' % conf.netName):
            # no way to make this efficient but matrices are small
            b_weighted_v_list = []
            for i, (bst_in, bst_out) in enumerate(
                    zip(range(0, batch_input_len, conf.INPUTLEN),
                        range(0, batch_output_len, conf.OUTPUTLEN))):
                with tf.name_scope('batch_%d' % i):
                    weighted_v_heads_list = []
                    for j, hst in enumerate(range(0, conf.NUMHID, conf.HEADSIZE)):
                        with tf.name_scope('head_%d' % j):
                            q_head = b_q_heads_tiled[bst_in:bst_in+conf.INPUTLEN, hst:hst+conf.HEADSIZE]
                            k_head = b_k_heads_tiled[bst_out:bst_out+conf.OUTPUTLEN, hst:hst+conf.HEADSIZE]
                            qk_head = tf.matmul(q_head, k_head, transpose_b=True) # (16, 16)

                            # add bias then calculate max to prevent overflow and 1.0 / 0.0
                            bias_br = b_bias_br[i] # (16, 16)
                            qk_bias_head = tf.add(qk_head, bias_br)
                            qk_max_head = tf.reduce_max(qk_bias_head, axis=1, keepdims=True) # (16, 1)
                            qk_max_br_head = tf.matmul(qk_max_head, ones_for_br) # (16, 16)
                            qk_mm_head = tf.subtract(qk_bias_head, qk_max_br_head)
                            qk_mm_exp_head = tf.exp(qk_mm_head) # (16, 16)
                            norm_factor = tf.reduce_sum(qk_mm_exp_head, axis=1, keepdims=True) # (16, 1)
                            norm_factor_rec = tf.reciprocal(norm_factor) # (16, 1)
                            norm_factor_rec_br = tf.matmul(norm_factor_rec, ones_for_br) # (16, 16)
                            weight_head = tf.multiply(qk_mm_exp_head, norm_factor_rec_br) # (16, 16)
                            v_t_head = b_v_t_heads_tiled[hst:hst+conf.HEADSIZE, bst_out:bst_out+conf.OUTPUTLEN] # (16, 64)
                            weighted_v_head = tf.matmul(weight_head, v_t_head, transpose_b=True) # (16, 64)
                            weighted_v_heads_list.append(weighted_v_head)
                    # merge heads
                    weighted_v = tf.concat(weighted_v_heads_list, axis=1) # (16, 1024)
                    b_weighted_v_list.append(weighted_v)
            b_weighted_v_tiled = tf.concat(b_weighted_v_list, axis=0, # (64, 1024)
                name='b_weighted_v_tiled')

        # (1024, 1024) * (1024, 64) --> (1024, 64), 88.88%
        output_t = tf.matmul(tr_kernel_t, b_weighted_v_tiled,
            transpose_b=True, name='%s/part3/output' % conf.netName) # (1024, 64)
        sess = tf.Session()
        b_att_t_np = sess.run(output_t, feed_dict={
            b_input_x_r: b_input_x_r_np,
            b_input_y_r: b_input_y_r_np,
            b_bias_br: b_bias_br_np})
        b_att_np = b_att_t_np.T.reshape(conf.BATCHSIZE, conf.INPUTLEN, conf.NUMHID)
        print(b_att_np, b_att_np.shape)
        error = b_att_ref_np.ravel() - b_att_np.ravel()
        print('rms error:', np.sqrt((error * error).mean()))

        input_data = {
            'input_x_r:0': b_input_x_r_np,
            'input_y_r:0': b_input_y_r_np,
            'input_bias_br:0': b_bias_br_np,
            }
        conf.gen_graph(output_t, input_data, need_freezing=False)
