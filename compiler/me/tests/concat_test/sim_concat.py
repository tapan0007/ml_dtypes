import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_concat
from collections import deque
import numpy as np
import tensorflow as tf
import re


class ConcatSim:
    def __init__(self,concat):
        self.concat = concat
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
        self.LoadMoveFilters()
        self.ComputeConcatInTF()
        for i in self.concat.waveops:
            self.SimMatMul(i)
            self.SimPool(i)
        return self.Compare()

    def Compare (self):
        with tf.Session() as sess:
            concat_golden = sess.run(self.concat_tf)
        concat_test = self.ExtractConcatTensor()
#        print ("golden = ",concat_golden)
#        print ("test = ", concat_test)
        result = np.array_equal(concat_golden, concat_test)
#        print (result)
        return result

    def ComputeConcatInTF(self):
        self.concat_tf = tf.concat(self.ifmap_tf, axis=3)

    def SimPool(self, waveop):
        if (waveop.__class__.__name__ == "PoolWaveOpInfo"):
            first = True
            if (waveop.src_is_psum == True):
                input_tensor = self.last_psum_updater
            # FIXME : Need to add more pool functionality such as AvgPool
            if (waveop.pool_func == "MaxPool"):
                pool_op = tf.nn.max_pool(\
                    input_tensor\
                    , [1, 1, 1, 1]\
                    , [1, 1, 1, 1]\
                    , 'SAME'\
                    , data_format='NHWC'\
                    , name=waveop.name)
                self.pool_id += 1
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
                        tf.convert_to_tensor(
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
        for ifmap in self.concat.ifmaps:
            np_data = np.load(ifmap.file_name)
            self.ofmap_channel_cnt += np_data.shape[1]
            # Converts NCHW to NHWC due to the limitation of tf.conv2d
            axis = (0, 2, 3, 1)
            np_data = np.transpose(np_data, axis)
            self.ifmap_dict[ifmap.waveop_name] = np_data
            self.ifmap_tf.append(
                tf.convert_to_tensor(np_data, self.concat.datatype)
            )

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
