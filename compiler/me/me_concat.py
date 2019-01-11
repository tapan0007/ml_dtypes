import math
import re
from collections import deque
#import me_pool
import me_common_ds
import numpy as np

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
    def __init__(self, start_pos, amt, abs_start_pos = 0):
        self.start_pos = start_pos
        self.abs_start_pos = abs_start_pos
        self.moving_amt = amt
        return

class Concat:
    # Currently, only C direction Concat is supported
    # FIXME : Need to support H and W direction Concat
    # ifmaps : Array of input feature map specs in FMAPSpec
    def __init__(self, ifmaps, num_output_channels, datatype):
        self.ifmaps = ifmaps 
#        print ("size of ifmaps = %d"%len(self.ifmaps))
        assert(num_output_channels > 0)
        self.num_output_channels = num_output_channels
        self.PE_ROW = 128
        self.PE_COL = 64
        #self.PE_ROW = 4
        #self.PE_COL = 4
        self.move_filters = deque()
        self.waveops = []
        self.datatype = datatype
        self.subtile_infos = dict()
        self.mfilters_for_one_move = []
        self.ifmaps_for_one_move = []
        self.ifmap_channel_ranges_for_one_move = []
        return
    @classmethod
    def init_from_file_params (cls, ifmap_file_params):
        ifmaps = cls.CreateFMAPSpecsFromFileParams(ifmap_file_params)
        num_output_channels = cls.SumIFMAPChannels(ifmap_file_params)
        datatype = ifmap_file_params[0].data_type
        return cls(ifmaps, num_output_channels, datatype)
    
    @classmethod
    def CreateFMAPSpecsFromFileParams (cls, ifmap_file_params):
        ifmaps = []
        cls.FMAPSpec2FileParams = dict()
        for ifmap_file_param in ifmap_file_params:
            N = ifmap_file_param.file_dims.dim["N"]
            C = ifmap_file_param.file_dims.dim["C"]
            H = ifmap_file_param.file_dims.dim["H"]
            W = ifmap_file_param.file_dims.dim["W"]
            fmap_dim = [N, C, H, W]
            file_name = ifmap_file_param.file_name
            ifmaps.append(me_common_ds.FMAPSpec(False,fmap_dim,file_name,""))
            cls.FMAPSpec2FileParams[ifmaps[-1]] = ifmap_file_param
        return ifmaps

    def SumIFMAPChannels(ifmap_file_params):
        ofmap_channel_cnt = 0
        for ifmap_file_param in ifmap_file_params:
            ofmap_channel_cnt += ifmap_file_param.file_dims.dim["C"]
        return ofmap_channel_cnt

    def print_graph(self):
        print("digraph G{")
        for i in self.waveops:
            for j in i.prev_waveops:
                if (i.__class__.__name__ == "MMWaveOpInfo"):
                    print("%s"%("\"" + j + "\"" + "->" + "\"" + i.name\
                        + "_" + str(i.start_tensor_calc)\
                        + "\""))
                else:
                    print("%s"%("\"" + j + "\"" + "->" + "\"" + i.name + "\""))
        print("}")

    def ConvertFMAPSpec2FMAPMovingRegion(self, ifmap, forward_move):
        if (forward_move == True):
            if (ifmap.start_mid == False):
                ifmap_region =\
                    FMAPMovingRegion(0, ifmap.C)
            else:
                ifmap_region =\
                    FMAPMovingRegion(int(self.PE_ROW / 2) - 1\
                                     ,ifmap.C\
                                     ,int(self.PE_ROW / 2) - 1\
                                    )
        else:
            if (ifmap.start_mid == False):
                ifmap_region =\
                    FMAPMovingRegion(ifmap.C - 1, ifmap.C)
            else:
                ifmap_region =\
                    FMAPMovingRegion(ifmap.C - 1 + int(self.PE_ROW / 2)\
                                     ,ifmap.C\
                                     ,ifmap.C - 1 + int(self.PE_ROW / 2)\
                                    )
        return ifmap_region

    def FirstOFMAPRegion (self, forward_move, tail):
        if (forward_move == True):
            if (self.num_output_channels < self.PE_COL):
                ofmap_region = FMAPMovingRegion(0, self.num_output_channels)
            else:
                ofmap_region = FMAPMovingRegion(0, self.PE_COL)
        else:
            if (tail):
                ofmap_region = FMAPMovingRegion(tail - 1, tail)
            else:
                ofmap_region = FMAPMovingRegion(self.PE_COL - 1, self.PE_COL)
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


    def ComputeStartTensorCalc (\
            self, num_channels_moved_sofar, forward_move, tail
                               ):
        start = True
        if (forward_move == True):
            if ((num_channels_moved_sofar % self.PE_COL) != 0):
                start = False
        else:
            if (num_channels_moved_sofar == 0):
                start = True
            elif (((num_channels_moved_sofar - tail) % self.PE_COL) != 0):
                start = False
        return start

    def PerformConcatDecomposition(self\
                                   , forward_move\
                                   , generate_weight_files = False):
        remaining_ifmaps = self.ifmaps.copy()
        remaining_ofmap_c = self.num_output_channels
        tail = self.num_output_channels % self.PE_COL
#        print ("tail = %d"%tail)
#        ofmap_region = FMAPMovingRegion(tail - 1, tail)
        ofmap_region = self.FirstOFMAPRegion(forward_move, tail)
#        ifmap = remaining_ifmaps.pop()
        ifmap = self.GetIFMAP(forward_move, remaining_ifmaps)
        ifmap_region = self.ConvertFMAPSpec2FMAPMovingRegion(ifmap,forward_move)
        ifmap_use_cnt = 0
        ofmap_move_cnt = 0
        pool_prev_ops = []
        num_channels_moved_sofar = 0;
        self.InitSubTileInfo()
        while remaining_ofmap_c > 0:
#            print ("num_channels_moved_sofar = %d"%num_channels_moved_sofar)
            start_tensor_calc =\
                    self.ComputeStartTensorCalc(num_channels_moved_sofar\
                        , forward_move, tail)
            if (ofmap_region.moving_amt == 0):
                #print ("len(pool_prev_ops) = %d ifmap_use_cnt = %d"\
                #       %(len(pool_prev_ops), ifmap_use_cnt))
                #print ("pool_prev_ops[-1] = %s"%pool_prev_ops[-1])
                if (len(pool_prev_ops) > 0):
                    pool = self.CreatePool(\
                        ifmap, pool_prev_ops, pool_prev_ops[-1]\
                        , ifmap_use_cnt - 1)
                    ofmap_c_start = ofmap_move_cnt * self.PE_COL
                    ofmap_c_end = ofmap_c_start + self.PE_COL - 1
                    self.AddSubTileInfo((ofmap_c_start, ofmap_c_end)
                                , [self.mfilters_for_one_move\
                                   , self.ifmaps_for_one_move\
                                   , self.ifmap_channel_ranges_for_one_move])
                    self.waveops.append(pool)
                self.InitSubTileInfo()
                pool_prev_ops = []
                ofmap_move_cnt += 1
                ofmap_region = self.GetOFMAPRegion(forward_move)
            if (ifmap_region.moving_amt == 0):
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
                if (ifmap_region.start_pos + ifmap_region.moving_amt\
                    >= self.PE_ROW):
                    moved_amt = self.PE_ROW - ifmap_region.start_pos
                else:
                    moved_amt = ifmap_region.moving_amt
                mfilter = self.ComputeMoveFilterSpec(\
                                                    ifmap_region.start_pos\
                                                    , ofmap_region.start_pos\
                                                    #, ifmap_region.moving_amt\
                                                    , moved_amt\
                                                    , forward_move\
                                                   )
                #moved_amt = ifmap_region.moving_amt
                ifmap_region_start_pos_before_update =ifmap_region.abs_start_pos
                if (forward_move == True):
                    ofmap_region.start_pos =\
                        ofmap_region.start_pos + moved_amt
                else:
                    ofmap_region.start_pos =\
                        ofmap_region.start_pos - ifmap_region.moving_amt
                ofmap_region.moving_amt =\
                    ofmap_region.moving_amt - moved_amt
                remaining_ofmap_c = remaining_ofmap_c - moved_amt
                num_channels_moved_sofar += moved_amt
                ifmap_region.moving_amt -= moved_amt
                ifmap_region.start_pos = (ifmap_region.start_pos + moved_amt)\
                  % self.PE_ROW
                ifmap_region.abs_start_pos =\
                  ifmap_region.abs_start_pos + moved_amt
            else:
                if (ifmap_region.start_pos + ofmap_region.moving_amt\
                    >= self.PE_ROW):
                    moved_amt = self.PE_ROW - ifmap_region.start_pos
                else:
                    moved_amt = ofmap_region.moving_amt
                mfilter = self.ComputeMoveFilterSpec(\
                                                    ifmap_region.start_pos\
                                                    , ofmap_region.start_pos\
                                                    , moved_amt\
                                                    , forward_move\
                                                   )
                #moved_amt = ofmap_region.moving_amt
                ifmap_region_start_pos_before_update =ifmap_region.abs_start_pos
                ifmap_region.moving_amt =\
                    ifmap_region.moving_amt - moved_amt
                if (forward_move == True):
                    ifmap_region.start_pos =\
                        (ifmap_region.start_pos + moved_amt) % self.PE_ROW
                    ifmap_region.abs_start_pos =\
                        (ifmap_region.abs_start_pos + moved_amt)
                else:
                    ifmap_region.start_pos =\
                        ifmap_region.start_pos - ofmap_region.moving_amt
                remaining_ofmap_c = remaining_ofmap_c - moved_amt
                num_channels_moved_sofar += moved_amt
                ofmap_region.moving_amt -= moved_amt
                ofmap_region.start_pos += moved_amt
            try:
                self.UpdateSubTileInfo(self.FMAPSpec2FileParams[ifmap]\
                                       , mfilter\
                                       , ifmap_region_start_pos_before_update\
                                       , moved_amt\
                                      )
            except:
                pass
            mfilter.file_name = self.NameFilter(
                ifmap.file_name, mfilter, ifmap_use_cnt)
            pool_prev_ops.extend(\
                self.CreateWaveOps(\
                    ifmap, mfilter, ifmap_use_cnt, start_tensor_calc\
                                  )
                                )
#            print("taemk::mfilter.file_name = %s"%mfilter.file_name)
            self.move_filters.append(mfilter)
            ifmap_use_cnt += 1
        if (len(pool_prev_ops) > 0):
            pool = self.CreatePool(ifmap, pool_prev_ops, pool_prev_ops[-1]\
                , ifmap_use_cnt - 1)
            self.waveops.append(pool)
            #self.UpdateSubTileInfo(ifmap\
            #                       , mfilter\
            #                       , ifmap_region\
            #                       , moved_amt\
            #                      )
            ofmap_c_start = ofmap_move_cnt * self.PE_COL
            if (tail == 0):
                ofmap_c_end = ofmap_c_start + self.PE_COL - 1
            else:
                ofmap_c_end = ofmap_c_start + tail - 1
            self.AddSubTileInfo((ofmap_c_start, ofmap_c_end)
                                , [self.mfilters_for_one_move\
                                   , self.ifmaps_for_one_move\
                                   , self.ifmap_channel_ranges_for_one_move])
        if (generate_weight_files == True):
            self.FilterWeightFileGeneration()
        return self.move_filters

    def UpdateSubTileInfo (self\
                           , ifmap\
                           , mfilter\
                           , ifmap_region_start_pos\
                           , moved_amt\
                          ):
        self.ifmaps_for_one_move.append(ifmap)
        self.mfilters_for_one_move.append(mfilter)
        self.ifmap_channel_ranges_for_one_move.append(\
                                                 (ifmap_region_start_pos\
                                                  , ifmap_region_start_pos\
                                                  + moved_amt - 1)\
                                                )
        return
    
    def InitSubTileInfo(self):
        self.mfilters_for_one_move = []
        self.ifmaps_for_one_move = []
        self.ifmap_channel_ranges_for_one_move = []
        return

    def AddSubTileInfo(self, ofmap_channel_range, subtile_info):
        self.subtile_infos[ofmap_channel_range] = subtile_info

    def GetSubTile(self, ofmap_channel_ranges):
        return self.subtile_infos[ofmap_channel_ranges]

    def PrintSubTileInfos(self):
        for (key, val) in self.subtile_infos.items():
            print (key)
            print ("len(mfilters_for_one_move) = %d"\
                   %(len(val[0])))
            #print ("len(ifmaps_for_one_move) = %d"\
            #       %(len(val[1])))
            #print ("len(ifmap_channel_ranges_for_one_move) = %d"\
            #       %(len(val[2])))
            for i in val[2]:
                print (i)
            print ("---")


    # Creates WaveOps for LDW and MatMul
    # Also creates dependency
    def CreateWaveOps(self, ifmap, mfilter, ifmap_use_cnt, start_tensor_calc):
        ops = []
        ldw = self.CreateLDW(mfilter)
        ops.append(ldw.name)
        self.waveops.append(ldw)
        prev_ops = [ldw.name, ifmap.waveop_name]
        mm = self.CreateMM(\
           ifmap, mfilter, prev_ops, ifmap_use_cnt, start_tensor_calc\
                          )
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
                + "_" + str(filter_spec.size) + "_concat_weight_CRSM.npy"
        return filter_name

    # ifmap : FMapSpec
    # weight : MoveFilterSpec
    # prev_waveops : an array of waveops that must be finished before current
    #                MM
    def CreateMM (self, ifmap, weight, prev_waveops, mmid, start_tensor_calc):
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
        mm = me_common_ds.MMWaveOpInfo(\
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
                          , start_tensor_calc\
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
        return me_common_ds.LDWWaveOpInfo(\
                   move_filter, [self.PE_ROW, 1, 1, self.PE_COL])

    def CreatePool (self, ifmap, prev_ops, input_tensor, poolid):
        ifmap_name_wo_extension = re.sub('\.npy',"", ifmap.file_name)
        name = ifmap_name_wo_extension + "_copy_" + str(poolid)
        return me_common_ds.PoolWaveOpInfo (\
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
                               , pool_func = "MaxPool"\
                               , pool_scale = "1.0"\
                               , prev_waveops = prev_ops\
                               , input_tensor = input_tensor\
                               , src_is_psum = True\
                               , dst_is_psum = False\
                               , name = name\
                               , ifmap = None
                              )

    def FilterWeightFileGeneration (self):
        # CRSM shape
        shape = [self.PE_ROW, 1, 1, self.PE_COL]
        for i in self.move_filters:
            data = np.zeros(shape, self.datatype)
            for j in range(i.size):
                data[i.start_loc[0] + j][0][0][i.start_loc[1] + j] = 1
            np.save(i.file_name, data)
        return

    def GiganticFilterWeightFileGeneration (self):
        # CRSM shape
        shape = [self.PE_ROW, 1, 1, self.PE_COL * len(self.move_filters)]
        data = np.zeros(shape, self.datatype)
        filter_cnt = 0
        for i in self.move_filters:
            for j in range(i.size):
                m_loc = filter_cnt * self.PE_COL + i.start_loc[1] + j
                data[i.start_loc[0] + j][0][0][m_loc] = 1
            filter_cnt += 1
        # FIXME : Need to name properly
        #np.save(.file_name, data)
        return
