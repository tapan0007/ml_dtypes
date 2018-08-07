import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_concat
import me_pool
from collections import deque
import numpy as np
import tensorflow as tf
import re


class PoolSim:
    def __init__(self,pool):
        self.pool = pool 
        self.ifmap_dict = dict()
        self.weight_dict = dict()
        self.tf_ops = []
        self.tf_dict = dict()
        self.pool_tf = []
        self.ifmap_tf = []
        self.last_mm = None
        self.mm_id = 0
        self.pool_id = 0
        self.last_psum_updater = None
        # FIXME : pool_ops container is only for Concat
        # to model SB. For generic functional simulation,
        # PSUM and SB modeling should be done properly
        self.pool_ops = []
        self.ofmap_channel_cnt = 0

    def run (self):
        self.LoadIFMAPs()
        self.CreateOFMAP(self.pool)
        self.ComputePoolInTF()
        for i in self.pool.waveops:
            self.SimMatMul(i)
            self.SimPool(i)
        #self.RunPoolTensor()
        return self.Compare()

    def Compare (self):
        with tf.Session() as sess:
            pool_golden = sess.run(self.pool_tf)
        pool_test = self.ifmap_dict[self.pool.ofmap.waveop_name]
        print ("golden = ")
        print(pool_golden)
        print ("test = ")
        print(pool_test)
        result = np.array_equal(pool_golden, pool_test)
        return result

    def ComputePoolInTF(self):
        input_tensor_tf = tf.convert_to_tensor(\
            self.ifmap_dict[self.pool.ifmap.waveop_name]
            , self.pool.datatype
            , name = "ComputePoolInTF_"+self.pool.ifmap.file_name)
        r = self.pool.R
        s = self.pool.S
        stride_x = self.pool.Th
        stride_y = self.pool.Tv
        self.pool_tf = tf.nn.pool(\
            input = input_tensor_tf\
            , window_shape = [r, s]\
            , pooling_type = "AVG"\
            , padding = 'SAME'\
            , strides = [stride_x, stride_y]\
            , name = "Pool_In_TensorFlow"\
            , data_format = 'NHWC')

    # NHWC format due to the limitation of Conv2D in TensorFlow
    def ReadIFMAPTensor (self, file_in_mem, waveop):
        src_start = waveop.src_start
        s_w = src_start[1]
        s_h = src_start[0]
        #s_c = src_start[2]
        # FIXME : Change np.float16 to data type of a waveop
        if (waveop.__class__.__name__ == "PoolWaveOpInfo"):
            th = waveop.src_z_step
            num_h = waveop.src_z_num
            tv = waveop.src_w_step
            num_v = waveop.src_w_num
            h = tv * (num_v - 1) + waveop.src_y_num
            w = th * (num_h - 1) + waveop.src_x_num
            c = waveop.ifmap.C
        else:
            s_c = src_start[2]
            h = waveop.src_y_num * waveop.src_y_step
            w = waveop.src_x_num * waveop.src_x_step
            c = waveop.src_z_num * waveop.src_z_step
        t = np.empty((1, h, w, c), np.float16)
        print("---ReadIFMAPTensor---")
        print("src_y_num = %d src_x_num = %d src_z_num = %d"\
              %(waveop.src_y_num, waveop.src_x_num, waveop.src_z_num))
        print("src_y_step = %d src_x_step = %d src_z_step = %d"\
              %(waveop.src_y_step, waveop.src_x_step, waveop.src_z_step))
        print("h = %d w = %d c = %d"%(h, w, c))
        print("s_w = %d s_h = %d"%(s_w, s_h))
        print ("file_in_mem[0].shape = ", file_in_mem[0].shape)
        t[0] = file_in_mem[0,s_h:(s_h+h),s_w:(s_w+w),:]
        print ("t[0].shape = ", t[0].shape)
        return t

    # file_in_mem : reference to file corresponding an output file
    #               Thus, it shouldn't be re-declared in this method
    def WriteOFMAPTensor (self, file_in_mem, ofmap, waveop):
        h = waveop.dst_y_num * waveop.dst_y_step
        w = waveop.dst_x_num * waveop.dst_x_step
        c = waveop.dst_z_num * waveop.dst_z_step
        d_h = waveop.dst_start[0]
        d_w = waveop.dst_start[1]
        print("---WriteOFMAPTensor---")
        print("h = %d w = %d c = %d d_w = %d d_h = %d"\
              %(h, w, c, d_w, d_h))
        print("ofmap.shape = ", ofmap.shape)
        # FIXME : Change np.float16 to data type of a waveop
        if (waveop.__class__.__name__ == "PoolWaveOpInfo"):
            c = waveop.ifmap.C
            if (waveop.dst_x_step == 1 and\
                waveop.dst_y_step == 1):
                file_in_mem[0, d_h:(d_h+h), d_w:(d_w+w), :] = ofmap
            else:
                h_o = 0
                w_o = 0
                for idx_c in range(0, c):
                    for idx_h in range(0, h, waveop.dst_y_step):
                        for idx_w in range(0, w, waveop.dst_x_step):
                            file_in_mem[0,d_h+idx_h,d_w+idx_w,idx_c] =\
                                ofmap[0,h_o,w_o,idx_c]
                            w_o += 1
                        h_o += 1
        else:
            d_c = waveop.dst_start[2]
            if (waveop.dst_x_step == 1 and\
                waveop.dst_y_step == 1 and\
                waveop.dst_z_step == 1):
                file_in_mem[0,d_h:d_h+h,d_w:d_w+w,:] = ofmap
            else:
                c_o = 0
                h_o = 0
                w_o = 0
                for idx_c in range(0, c, waveop.dst_z_step):
                    for idx_h in range(0, h, waveop.dst_y_step):
                        for idx_w in range(0, w, waveop.dst_x_step):
                            file_in_mem[0,d_h+idx_h,d_w+idx_w,d_c+idx_c] =\
                                ofmap[0,h_o,w_o,c_o]
                            w_o += 1
                        h_o += 1
                    c_o += 1
        return

    def SimPool(self, waveop):
        if (waveop.__class__.__name__ == "PoolWaveOpInfo"):
            first = True
            if (waveop.src_is_psum == True):
                input_tensor_tf = self.last_psum_updater
            else:
                input_tensor = self.ReadIFMAPTensor(\
                    self.ifmap_dict[waveop.ifmap.waveop_name], waveop)
                input_tensor_tf = tf.convert_to_tensor(\
                    input_tensor\
                    , self.pool.datatype\
                    , name=waveop.name)
                self.tf_ops.append(input_tensor_tf)
            # FIXME : Need to add more pool functionality such as AvgPool
            if (waveop.pool_func == "MaxPool"):
                pooling_type = "MAX"
            # AvgPool
            else:
                pooling_type = "AVG"
            self.pool_id += 1
            r = waveop.src_x_num
            s = waveop.src_y_num
            stride_x = waveop.src_z_step
            stride_y = waveop.src_w_step
            if (stride_x > s): stride_x = s
            if (stride_y > r): stride_y = r
            print("---SimPool---")
            print("r = %d s = %d stride_x = %d stride_y = %d"\
                  %(r, s, stride_x, stride_y))
            print("input_tensor.shape = ", input_tensor.shape)
            pool_op = tf.nn.pool(\
                input = input_tensor_tf\
                , window_shape = [s, r]\
                , pooling_type = pooling_type\
                , padding = 'VALID'\
                , strides = [stride_x, stride_y]\
                , name = waveop.name\
                , data_format = 'NHWC')
            with tf.Session() as sess:
                p = sess.run(pool_op)
            self.WriteOFMAPTensor (self.ifmap_dict[self.pool.ofmap.waveop_name]\
                                   , p, waveop)
            if (waveop.dst_is_psum == True):
                self.last_psum_updater = pool_op
            self.tf_ops.append(pool_op)
            self.tf_dict[waveop.name] = pool_op
            self.pool_ops.append(pool_op)

    def SimMatMul(self, waveop):
        if (waveop.__class__.__name__ == "MMWaveOpInfo"):
            for j in waveop.prev_waveops:
                if (re.search(r'concat_weight_CRSM', j)):
                    weight = self.weight_dict[j]
                    weight_tf =\
                        tf.convert_to_tensor(\
                            weight\
                            , self.concat.datatype\
                            , name = j\
                        )
                    self.tf_ops.append(weight_tf)
                else:
                    image = self.ifmap_dict[j]
                    if (image.shape[3] != self.concat.PE_ROW):
                        shape = [image.shape[0]\
                                 , image.shape[1], image.shape[2]\
                                 , self.concat.PE_ROW - image.shape[3]]
                        zero_extension = np.zeros(shape)
                        image = np.concatenate((image, zero_extension), 3)
                    image_tf = \
                        tf.convert_to_tensor(\
                            image\
                            , self.concat.datatype\
                            , name = j
                                            )
                    self.tf_ops.append(image_tf)
            mm_op = tf.nn.conv2d(image_tf, weight_tf,\
                [1, 1, 1, 1], "VALID", name = waveop.name)
            self.tf_ops.append(mm_op)
            self.tf_dict[waveop.name] = mm_op
            if (waveop.start_tensor_calc != True):
                n = (waveop.name + "_psum_" + str(self.mm_id))
                add_tf = tf.add(\
                                self.last_mm\
                                , mm_op\
                                , name = n\
                               )
                self.last_mm = add_tf
                self.tf_ops.append(add_tf)
                self.tf_dict[n] = add_tf
                self.last_psum_updater = add_tf
            else:
                self.last_mm = mm_op
                self.last_psum_updater = mm_op
            self.mm_id += 1
#            print("image_tf.name = %s"%image_tf.name)
#            print("weight_tf.name = %s"%weight_tf.name)
#            print("mm_op.name = %s"%mm_op.name)


    def LoadIFMAPs(self):
#        for ifmap in self.pool.ifmaps:
        ifmap = self.pool.ifmap
        np_data = np.load(ifmap.file_name)
        self.ofmap_channel_cnt += np_data.shape[1]
        # Converts NCHW to NHWC due to the limitation of tf.conv2d
        axis = (0, 2, 3, 1)
        np_data = np.transpose(np_data, axis)
        self.ifmap_dict[ifmap.waveop_name] = np_data
        self.ifmap_tf.append(
            tf.convert_to_tensor(np_data, self.pool.datatype)
        )

    def CreateOFMAP(self, waveop):
        # ofmap in FMAPSpec
        if (hasattr(waveop, "ofmap")):
            n = waveop.ofmap.N
            c = waveop.ofmap.C
            h = waveop.ofmap.H
            w = waveop.ofmap.W
            self.ifmap_dict[waveop.ofmap.waveop_name] = np.empty([n, h, w, c])

    def LoadMoveFilters (self):
        for mf in self.concat.move_filters:
            data = np.load(mf.file_name)
            # Converts CRSM shape to RSCM shape since it is the only
            # weight shape that tf.conv2d accepts
            axis = (1, 2, 0, 3)
            datat = np.transpose(data, axis)
            #print("transposed weight shape = ", datat.shape)
            #print("--%s"%(mf.file_name))
            self.weight_dict[(mf.file_name + "_0")] = datat

################################################################################
#Concat Specific Methods:
#    These may become obsolete when a generic functional simulator is put in
#    place.
################################################################################

    # FIXME : Note that this is only for Concat operation
    #         Need to have a way to extract a certain portion of SB
    #         and make it a tensor
    def ExtractConcatTensor (self):
        first = True
        with tf.Session() as sess:
            for p in self.pool_ops:
                if (first == True):
                    result = sess.run(p)
                    first = False
                else:
                    result = np.concatenate((result, sess.run(p)), axis=3)
#        print (result)
        if (result.shape[3] > self.ofmap_channel_cnt):
            result = result[:, :, :, 0:(self.ofmap_channel_cnt)]
#        print (result)
        return result
    def RunPoolTensor (self):
        with tf.Session() as sess:
            print("Golden = ")
            print (sess.run(self.pool_tf))
        return
