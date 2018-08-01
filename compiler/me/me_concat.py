import math
import re
from collections import deque
import me_pool

class LDWWaveOpInfo():
    def __init__(self, move_filter, shape_in_crsm):
        self.ref_file = move_filter.file_name
        self.name = move_filter.file_name + "_0"
        self.ref_file_format = "CRSM"
        # default shape : [128, 1, 1, 64]
        self.ref_file_shape = shape_in_crsm
        self.prev_waveops = []

    def print_prev_ops(self):
        return

class MMWaveOpInfo(me_pool.WaveOpInfo):
    #ifmap : FMapSpec
    def __init__(self\
                 , src_x_num\
                 , src_x_step\
                 , src_y_num\
                 , src_y_step\
                 , src_z_num\
                 , src_z_step\
                 , src_w_num\
                 , src_w_step\
                 , dst_x_num\
                 , dst_x_step\
                 , dst_y_num\
                 , dst_y_step\
                 , dst_z_num\
                 , dst_z_step\
                 , src_start\
                 , dst_start\
                 , num_row_partitions\
                 , num_col_partitions\
                 , stride_x\
                 , stride_y\
                 , ifmap\
                 , prev_waveops\
                 , name\
                ):
        me_pool.WaveOpInfo.__init__(\
                                    self\
                                    , src_x_num\
                                    , src_x_step\
                                    , src_y_num\
                                    , src_y_step\
                                    , src_z_num\
                                    , src_z_step\
                                    , src_w_num\
                                    , src_w_step\
                                    , dst_x_num\
                                    , dst_x_step\
                                    , dst_y_num\
                                    , dst_y_step\
                                    , dst_z_num\
                                    , dst_z_step\
                                    , src_start\
                                    , dst_start\
                                   )
        self.num_row_partitions = num_row_partitions
        self.num_col_partitions = num_col_partitions
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.ifmap = ifmap
        self.prev_waveops = prev_waveops
        self.name = name

    def print_prev_ops(self):
        for i in self.prev_waveops:
            print (i)

class PoolWaveOpInfo(me_pool.WaveOpInfo):
    def __init__(self\
                 , src_x_num\
                 , src_x_step\
                 , src_y_num\
                 , src_y_step\
                 , src_z_num\
                 , src_z_step\
                 , src_w_num\
                 , src_w_step\
                 , dst_x_num\
                 , dst_x_step\
                 , dst_y_num\
                 , dst_y_step\
                 , dst_z_num\
                 , dst_z_step\
                 , src_start\
                 , dst_start\
                 , pool_frequency\
                 , pool_func\
                 , pool_scale\
                 , prev_waveops\
                 , name\
                ):
        me_pool.WaveOpInfo.__init__(\
                                    self\
                                    , src_x_num\
                                    , src_x_step\
                                    , src_y_num\
                                    , src_y_step\
                                    , src_z_num\
                                    , src_z_step\
                                    , src_w_num\
                                    , src_w_step\
                                    , dst_x_num\
                                    , dst_x_step\
                                    , dst_y_num\
                                    , dst_y_step\
                                    , dst_z_num\
                                    , dst_z_step\
                                    , src_start\
                                    , dst_start\
                                   )
        self.pool_frequency = pool_frequency
        self.pool_func = pool_func
        self.pool_scale = pool_scale
        self.prev_waveops = prev_waveops
        self.name = name

    def print_prev_ops(self):
        for i in self.prev_waveops:
            print (i)

class MoveFilterSpec:
    def __init__ (self, start_loc, size, file_name):
        self.start_loc = start_loc
        self.size = size
        self.file_name = file_name
        return
    def __init__ (self, start_loc, size):
        self.start_loc = start_loc
        self.size = size
        self.file_name = ""
        return
    def print(self):
        print(self.start_loc, ", size = %d"%self.size)
    def __ne__(self, f):
        return ((self.start_loc != f.start_loc) |\
                (self.size != f.size))

# moving direction is toward 0-th channel when move is done backward
# |----|
# |  0 |
# |----|------------|
# |  1 |   /\
# |----|    |
# |  2 |    | amt (moving amount)
# |----|    |    moving direction
# |  3 |    |   /\
#    .     \/   |
#    .  --------+---| start_pos
#    .
# |----|
# | 63 |
# |----|
class FMAPMovingRegion:
    def __init__(self, start_pos, amt):
        self.start_pos = start_pos
        self.moving_amt = amt
        return
class FMAPSpec:
    # fmap_dim : NCHW format
    def __init__(self, start_mid, fmap_dim, file_name, waveop_name):
        self.start_mid = start_mid
#        self.channel_num = channel_num
#        self.fmap_dim = fmap_dim
        self.N = fmap_dim[0]
        self.C = fmap_dim[1]
        self.H = fmap_dim[2]
        self.W = fmap_dim[3]
        self.file_name = file_name
        self.waveop_name = waveop_name
        return

class Concat:
    # Currently, only C direction Concat is supported
    # FIXME : Need to support H and W direction Concat
    # ifmaps : Array of input feature map specs in FMAPSpec
    def __init__(self, ifmaps, num_output_channels):
        self.ifmaps = ifmaps 
        assert(num_output_channels > 0)
        self.num_output_channels = num_output_channels
        self.PE_ROW = 128
        self.PE_COL = 64
        self.move_filters = deque()
        self.waveops = []
        return
    def print_graph(self):
        print("digraph G{")
        for i in self.waveops:
            for j in i.prev_waveops:
                print("%s"%("\"" + j + "\"" + "->" + "\"" + i.name + "\""))
        print("}")

    def ConvertFMAPSpec2FMAPMovingRegion(self, ifmap, forward_move):
        if (forward_move == True):
            if (ifmap.start_mid == False):
                ifmap_region =\
                    FMAPMovingRegion(0, ifmap.C)
            else:
                ifmap_region =\
                    FMAPMovingRegion(int(self.PE_ROW / 2) - 1,ifmap.C)
        else:
            if (ifmap.start_mid == False):
                ifmap_region =\
                    FMAPMovingRegion(ifmap.C - 1, ifmap.C)
            else:
                ifmap_region =\
                    FMAPMovingRegion(ifmap.C - 1 + int(self.PE_ROW / 2),ifmap.C)
        return ifmap_region

    def FirstOFMAPRegion (self, forward_move, tail):
        if (forward_move == True):
            if (self.num_output_channels < self.PE_COL):
                ofmap_region = FMAPMovingRegion(0, self.num_output_channels)
            else:
                ofmap_region = FMAPMovingRegion(0, self.PE_COL)
        else:
            ofmap_region = FMAPMovingRegion(tail - 1, tail)
        return ofmap_region

    def GetOFMAPRegion (self, forward_move):
        if (forward_move == True):
            ofmap_region = FMAPMovingRegion(0, self.PE_COL)
        else:
            ofmap_region = FMAPMovingRegion(self.PE_COL - 1, self.PE_COL)
        return ofmap_region

    def GetIFMAP (self, forward_move, remaining_ifmaps):
        if (forward_move == True):
            ifmap = remaining_ifmaps.pop(0)
        else:
            ifmap = remaining_ifmaps.pop()
        return ifmap


    def PerformConcatDecomposition(self, forward_move):
        remaining_ifmaps = self.ifmaps
        remaining_ofmap_c = self.num_output_channels
        tail = self.num_output_channels % self.PE_COL
#        print ("tail = %d"%tail)
#        ofmap_region = FMAPMovingRegion(tail - 1, tail)
        ofmap_region = self.FirstOFMAPRegion(forward_move, tail)
#        ifmap = remaining_ifmaps.pop()
        ifmap = self.GetIFMAP(forward_move, remaining_ifmaps)
        ifmap_region = self.ConvertFMAPSpec2FMAPMovingRegion(ifmap,forward_move)
        ifmap_use_cnt = 0
        pool_prev_ops = []
        while remaining_ofmap_c > 0:
            if (ofmap_region.moving_amt == 0):
                pool = self.CreatePool(ifmap, pool_prev_ops, ifmap_use_cnt - 1)
                self.waveops.append(pool)
                pool_prev_ops = []
#                ofmap_region = FMAPMovingRegion(self.PE_COL - 1, self.PE_COL)
                ofmap_region = self.GetOFMAPRegion(forward_move)
            if (ifmap_region.moving_amt == 0):
#                ifmap = remaining_ifmaps.pop()
                ifmap = self.GetIFMAP(forward_move, remaining_ifmaps)
                ifmap_use_cnt = 0
                ifmap_region =\
                    self.ConvertFMAPSpec2FMAPMovingRegion(ifmap,forward_move)
#            print("remaining_ofmap_c = ", remaining_ofmap_c)
#            print("ofmap_region.moving_amt = ", ofmap_region.moving_amt)
#            print("ofmap_region.start_pos = ", ofmap_region.start_pos)
#            print("ifmap_region.moving_amt = ", ifmap_region.moving_amt)
#            print("ifmap_region.start_pos = ", ifmap_region.start_pos)
            if (ofmap_region.moving_amt >= ifmap_region.moving_amt):
                mfilter = self.ComputeMoveFilterSpec(\
                                                    ifmap_region.start_pos\
                                                    , ofmap_region.start_pos\
                                                    , ifmap_region.moving_amt\
                                                    , forward_move\
                                                   )
                if (forward_move == True):
                    ofmap_region.start_pos =\
                        ofmap_region.start_pos + ifmap_region.moving_amt
                else:
                    ofmap_region.start_pos =\
                        ofmap_region.start_pos - ifmap_region.moving_amt
                ofmap_region.moving_amt =\
                    ofmap_region.moving_amt - ifmap_region.moving_amt
                remaining_ofmap_c = remaining_ofmap_c - ifmap_region.moving_amt
                ifmap_region.moving_amt = 0
            else:
                mfilter = self.ComputeMoveFilterSpec(\
                                                    ifmap_region.start_pos\
                                                    , ofmap_region.start_pos\
                                                    , ofmap_region.moving_amt\
                                                    , forward_move\
                                                   )
                ifmap_region.moving_amt =\
                    ifmap_region.moving_amt - ofmap_region.moving_amt
                if (forward_move == True):
                    ifmap_region.start_pos =\
                        ifmap_region.start_pos + ofmap_region.moving_amt
                else:
                    ifmap_region.start_pos =\
                        ifmap_region.start_pos - ofmap_region.moving_amt
                remaining_ofmap_c = remaining_ofmap_c - ofmap_region.moving_amt
                ofmap_region.moving_amt = 0
            mfilter.file_name = self.NameFilter(
                ifmap.file_name, mfilter, ifmap_use_cnt)
            pool_prev_ops.extend(\
                self.CreateWaveOps(ifmap, mfilter, ifmap_use_cnt)
                                )
#            print("taemk::mfilter.file_name = %s"%mfilter.file_name)
            self.move_filters.append(mfilter)
            ifmap_use_cnt += 1
        if (len(pool_prev_ops) > 0):
            pool = self.CreatePool(ifmap, pool_prev_ops, ifmap_use_cnt - 1)
            self.waveops.append(pool)
        return self.move_filters

    # Creates WaveOps for LDW and MatMul
    # Also creates dependency
    def CreateWaveOps(self, ifmap, mfilter, ifmap_use_cnt):
        ops = []
        ldw = self.CreateLDW(mfilter)
        ops.append(ldw.name)
        self.waveops.append(ldw)
        prev_ops = [ldw.name, ifmap.waveop_name]
        mm = self.CreateMM(ifmap, mfilter, prev_ops, ifmap_use_cnt)
        ops.append(mm.name)
        self.waveops.append(mm)
        #pool = self.CreatePool(ifmap, [mm.name], ifmap_use_cnt)
        #self.waveops.append(pool)
        return ops

    def NameFilter (self, ifmap_name, filter_spec, ifmap_use_cnt):
        ifmap_name_wo_extension = re.sub('\.npy', '', ifmap_name)
        filter_name = ifmap_name_wo_extension\
                + "_" + str(ifmap_use_cnt)\
                + "_" + str(filter_spec.start_loc[0])\
                + "_" + str(filter_spec.start_loc[1])\
                + "_" + str(filter_spec.size) + "_concat_weight.npy"
        return filter_name

    # ifmap : FMapSpec
    # weight : MoveFilterSpec
    # prev_waveops : an array of waveops that must be finished before current
    #                MM
    def CreateMM (self, ifmap, weight, prev_waveops, mmid):
        src_x_num = ifmap.W
        src_x_step = 1
        src_y_num = ifmap.H
        src_y_step = ifmap.W
        src_z_num = ifmap.C
        src_z_step = ifmap.H * ifmap.W
        src_w_num = 1
        src_w_step = src_z_step * ifmap.C
        dst_x_num = ifmap.W
        dst_x_step = 1
        dst_y_num = ifmap.H
        dst_y_step = ifmap.W
        dst_z_num = 1
        dst_z_step = ifmap.H * ifmap.W
        # FIXME : This should be optimized so that we do not
        #         generate unnecessary zeros in filter weights
        num_row_partitions = self.PE_ROW
        num_col_partitions = self.PE_COL
        stride_x = 1
        stride_y = 1
        ifmap_name_wo_extension = re.sub('\.npy',"", ifmap.file_name)
        name = ifmap_name_wo_extension + "_MatMul_" + str(mmid)
        mm = MMWaveOpInfo(\
                          src_x_num\
                          , src_x_step\
                          , src_y_num\
                          , src_y_step\
                          , src_z_num\
                          , src_z_step\
                          , src_w_num\
                          , src_w_step\
                          , dst_x_num\
                          , dst_x_step\
                          , dst_y_num\
                          , dst_y_step\
                          , dst_z_num\
                          , dst_z_step\
                          , (0, 0)\
                          , (0, 0)\
                          , num_row_partitions\
                          , num_col_partitions\
                          , stride_x\
                          , stride_y\
                          , ifmap\
                          , prev_waveops\
                          , name\
                         )
        return mm

    def ComputeMoveFilterSpec (self\
                               , ifmap_start_pos\
                               , ofmap_start_pos\
                               , moving_amt\
                               , forward_move
                              ):
        if (forward_move == True):
            start_row = ifmap_start_pos
            start_col = ofmap_start_pos
        else:
            start_row = ifmap_start_pos - moving_amt + 1
            start_col = ofmap_start_pos - moving_amt + 1
        return MoveFilterSpec((start_row, start_col), moving_amt)

    def CreateLDW (self, move_filter):
        return LDWWaveOpInfo(move_filter, [128, 1, 1, 64])

    def CreatePool (self, ifmap, prev_ops, poolid):
        ifmap_name_wo_extension = re.sub('\.npy',"", ifmap.file_name)
        name = ifmap_name_wo_extension + "_copy_" + str(poolid)
        return PoolWaveOpInfo (\
                               src_x_num = ifmap.W\
                               , src_x_step = 1\
                               , src_y_num = ifmap.H\
                               , src_y_step = ifmap.W\
                               , src_z_num = 64\
                               , src_z_step = ifmap.H * ifmap.W\
                               , src_w_num = 1\
                               , src_w_step = ifmap.H * ifmap.W * 64\
                               , dst_x_num = ifmap.W\
                               , dst_x_step = 1\
                               , dst_y_num = ifmap.H\
                               , dst_y_step = ifmap.W\
                               , dst_z_num = 64\
                               , dst_z_step = ifmap.H * ifmap.W\
                               , src_start = (0, 0)\
                               , dst_start = (0, 0)\
                               , pool_frequency = 1\
                               , pool_func = "MaxPool"\
                               , pool_scale = "1.0"\
                               , prev_waveops = prev_ops\
                               , name = name\
                              )
