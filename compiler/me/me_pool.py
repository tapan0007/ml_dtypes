import math
import me_common_ds
import me_utils
import numpy as np

###############################################################################
# Filter Window Dimension
#         S
#    +---------+
#    |         |
#   R|         |
#    |         |
#    +---------+
###############################################################################

class Pool_WaveOps:
  def __init__(self):
    self.ltcp_ops = []
    self.rtcp_ops = []
    self.lbcp_ops = []
    self.rbcp_ops = []
    self.uhep_ops = []
    self.bhep_ops = []
    self.lvep_ops = []
    self.rvep_ops = []
    self.cp_ops = []
    return

class Pool:
  def __init__(self, ifmap, pool_window, padding, pool_func, src_is_psum\
              , dst_is_psum, ofmap, dtype):
    self.ifmap = ifmap
    self.pool_window = pool_window
    self.padding = padding
    self.H = self.ifmap.H
    self.W = self.ifmap.W
    self.R = self.pool_window.R
    self.S = self.pool_window.S
    self.pW = self.padding.pW
    self.pE = self.padding.pE
    self.pS = self.padding.pS
    self.pN = self.padding.pN
    self.Th = self.pool_window.Th
    self.Tv = self.pool_window.Tv
    self.pool_func = pool_func
    self.src_is_psum = src_is_psum
    self.dst_is_psum = dst_is_psum
    self.waveops = []
    self.ofmap = ofmap
    self.datatype = dtype
    def getMinimum(a):
      int_a = int(a)
      return (int_a + 1)

    self.t = getMinimum((self.pN + self.H - self.R) / self.Tv)
    self.o = int((self.pN + self.pS + self.H - self.R) / self.Tv - self.t)
    self.u = getMinimum((self.pW + self.W - self.S) / self.Th)
    self.q = int((self.pW + self.W + self.pE - self.S) / self.Th - self.u)
    self.ofmap_H = int((self.H + self.pS + self.pN - self.R) / self.Tv +  1)
    self.ofmap_W = int((self.W + self.pE + self.pW - self.S) / self.Th +  1)
    
    # l :
    # Gap between left edge of IFMAP and that of the left most pool window
    # for horizontal edge pool
    # -------------------------------------- Padding boundary
    # |     ^
    # |     | pN  ***************************** Superposed pool windows
    # |    \/     *
    # |<--->------*-------------------------- IFMAP boundary
    # |  pW |     *
    # |     |<--->*
    # |     |  l  *
    self.l = math.ceil(self.pW / self.Th) * self.Th - self.pW
    # a : The number of pool windows for a single horizontal edge pool
    self.a = math.floor((self.W - self.l - self.S) / self.Th)
    self.m = math.ceil(self.pN / self.Tv) * self.Tv - self.pN
    self.b = math.floor((self.H - self.m - self.R) / self.Tv)
    self.pool_waveops = Pool_WaveOps()
    return

  @classmethod
  def init_from_rectangles(cls\
                           , padded_ifmap_tile_rect\
                           , ifmap_tile_rect\
                           , ofmap_tile_rect\
                           , org_pWN\
                           , pool_window\
                           , pool_stride\
                          ):
      ifmap_h = ifmap_tile_rect.get_width_height().y
      ifmap_w = ifmap_tile_rect.get_width_height().x
      ifmap = me_common_ds.FMAPDim(ifmap_h, ifmap_w)
      #print ("ifmap_dim : ", ifmap)
      filterspec = me_common_ds.FilterSpec(\
        pool_window.y, pool_window.x, pool_stride.x, pool_stride.y)
      lower_offset = ifmap_tile_rect.get_offset_from(padded_ifmap_tile_rect)
      upper_offset = padded_ifmap_tile_rect.upper - ifmap_tile_rect.upper
      padding = me_common_ds.PaddingSpec(\
        abs(lower_offset.y), abs(upper_offset.y)\
        , abs(lower_offset.x), abs(upper_offset.x))
      cls.padded_ifmap_tile_rect = padded_ifmap_tile_rect
      cls.ifmap_tile_rect = ifmap_tile_rect
      cls.ofmap_tile_rect = ofmap_tile_rect
      cls.pool_window_dim2d = pool_window
      cls.pool_stride_dim2d = pool_stride
      cls.org_pWN = org_pWN
      assert(((padded_ifmap_tile_rect.lower.y + org_pWN.y)%filterspec.Tv) == 0)
      assert(((padded_ifmap_tile_rect.lower.x + org_pWN.x)%filterspec.Th) == 0)
      cls.ofmap_offset_y =\
              int((padded_ifmap_tile_rect.lower.y+org_pWN.y)/filterspec.Tv)
      cls.ofmap_offset_x =\
              int((padded_ifmap_tile_rect.lower.x+org_pWN.x)/filterspec.Th)
      print("ofmap_offset_x = %d ofmap_offset_y = %d"\
            %(cls.ofmap_offset_x, cls.ofmap_offset_y))
      return cls(
          ifmap = ifmap\
          , pool_window = filterspec\
          , padding = padding\
          , pool_func = ""\
          , src_is_psum = False\
          , dst_is_psum = False\
          , ofmap = ifmap\
          , dtype = np.float16\
      )

  # CP : Corner Pool
  # HV : Horizontal and Vertical
  # LTCP : Left Top Corner Pool
  # RTCP : Right Top Corner Pool
  # LBCP : Left Bottom Corner Pool
  # RBCP : Right Bottom Corner Pool
  # UHEP : Upper Horizontal Edge Pool
  # BHEP : Bottom Horizontal Edge Pool
  # LVEP : Lef Veritical Edge Pool
  # RVEP : Right Vertical Edge Pool
  # CP : Center Pool
  def CP_ComputeHVMoveCount (self, corner_type):
    if (corner_type == "LTCP"):
      nh = int(math.ceil(self.pW / self.Th))
      nv = int(math.ceil(self.pN / self.Tv))
    elif (corner_type == "RTCP"):
      nv = int(math.ceil(self.pN / self.Tv))
      nh = self.q + 1
    elif (corner_type == "LBCP"):
      nv = self.o + 1
      nh = int(math.ceil(self.pW / self.Th))
    else:
      nv = self.o + 1
      nh = self.q + 1
#    print ("nh = %d, nv = %d"%(nh, nv))
    return (nh, nv)
  def EP_ComputeHVMoveCount (self, edge_type):
    nh = 1
    nv = 1
    if (edge_type == "UHEP"):
      nv = int(math.ceil(self.pN / self.Tv))
    elif (edge_type == "BHEP"):
      nv = self.o + 1
    elif (edge_type == "LVEP"):
      nh = int(math.ceil(self.pW / self.Th))
    else:
      nh = self.q + 1
#    print ("nh = %d, nv = %d"%(nh, nv))
    return (nh, nv)

  # Variables : src_x_num, src_y_num, src_start and dst_start
  # i : i-th horizontal move
  # j : j-th vertical move
  def ComputeCPVariables (self, corner_type, i, j):
    if (corner_type == "LTCP"):
      src_start = (0, 0)
      dst_start = (j, i)
      src_x_num = self.S - self.pW + i * self.Th
      src_y_num = self.R - self.pN + j * self.Tv
    elif (corner_type == "RTCP"):
      src_x_num = self.W + self.pW - (self.u + i) * self.Th
      src_y_num = self.R - self.pN + j * self.Tv
      src_start = (0, self.u * self.Th - self.pW + i * self.Th)
      #dst_start = (j, self.W - (self.q + 1) + i)
      dst_start = (j, self.ofmap_W - (self.q + 1) + i)
    elif (corner_type == "LBCP"):
      src_x_num = self.S - self.pW + i * self.Th
      src_y_num = self.H + self.pN - (self.t + j) * self.Tv
      src_start = (self.t * self.Tv - self.pN + j * self.Tv, 0)
      #dst_start = (self.H - (self.o + 1) + j, i)
      dst_start = (self.ofmap_H - (self.o + 1) + j, i)
    else:
      src_x_num = self.W + self.pW - (self.u + i) * self.Th
      src_y_num = self.H + self.pN - (self.t + j) * self.Tv
      src_start = (self.t * self.Tv - self.pN + j * self.Tv\
          , self.u * self.Th - self.pW + i * self.Th)
      #dst_start = (self.H - (self.o + 1) + j, self.W - (self.q + 1) + i)
      #dst_start = (self.ofmap_H - (self.o + 1) + j, self.W - (self.q + 1) + i)
      dst_start = (self.ofmap_H - (self.o + 1) + j\
                   , self.ofmap_W - (self.q + 1) + i)
    return (src_x_num, src_y_num, src_start, dst_start)

  def ComputeCP (self, corner_type):
    (nh, nv) = self.CP_ComputeHVMoveCount(corner_type)
    src_x_step = 1
    src_y_step = 1 #self.W
    src_z_num = 1
    src_z_step = self.Th
    src_w_num = 1
    src_w_step = self.Tv #self.W
    dst_x_num = 1
    dst_x_step = 1
    dst_y_num = 1
    dst_y_step = 1 #self.W
    dst_z_num = 1
    dst_z_step = 1 #(self.W * self.H)
    for i in range(nh):
      for j in range(nv):
        (src_x_num, src_y_num, src_start, dst_start) = \
          self.ComputeCPVariables(corner_type, i, j)
        waveop = me_common_ds.WaveOpInfo(\
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
            , src_start\
            , dst_start\
            )
        if (corner_type == "LTCP"):
          self.pool_waveops.ltcp_ops.append(waveop)
        elif (corner_type == "RTCP"):
          self.pool_waveops.rtcp_ops.append(waveop)
        elif (corner_type == "LBCP"):
          self.pool_waveops.lbcp_ops.append(waveop)
        else:
          self.pool_waveops.rbcp_ops.append(waveop)
    return
  def ComputeEPVariables(self, edge_type, i):
    if (edge_type == "UHEP" or edge_type == "BHEP"):
      src_x_num = self.S
      if (edge_type == "UHEP"):
        src_y_num = self.R - self.pN + i * self.Tv
        src_start = (0, math.ceil(self.pW / self.Th) * self.Th - self.pW)
        dst_start = (i, math.ceil(self.pW / self.Th))
      else:
        src_y_num = self.H + self.pN - (self.t + i) * self.Tv
        src_start = (self.t * self.Tv - self.pN + i * self.Tv\
            , math.ceil(self.pW / self.Th) * self.Th - self.pW)
        #dst_start = (self.H - (self.o + 1) + i, math.ceil(self.pW / self.Th))
        dst_start = (self.ofmap_H - (self.o + 1) + i\
                     , math.ceil(self.pW / self.Th))
    else:
      src_y_num = self.R
      if (edge_type == "LVEP"):
        src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN, 0)
        dst_start = (math.ceil(self.pN / self.Tv), i)
        src_x_num = self.S - self.pW + i * self.Th
      else:
        src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN\
            , self.u * self.Th - self.pW + i * self.Th)
        #dst_start = (math.ceil(self.pN / self.Tv), self.W - (self.q + 1) + i)
        dst_start = (math.ceil(self.pN / self.Tv)\
                     , self.ofmap_W - (self.q + 1) + i)
        src_x_num = self.W + self.pW - (self.u + i) * self.Th
    return (src_x_num, src_y_num, src_start, dst_start)
  # EP : Edge Pool
  def ComputeEP (self, edge_type):
    (nh, nv) = self.EP_ComputeHVMoveCount(edge_type)
    if (edge_type == "UHEP" or edge_type == "BHEP"):
      src_z_num = self.a + 1
      src_w_num = 1
      dst_x_num = self.a + 1
      dst_y_num = 1
      num_pools = nv
    else:
      src_z_num = 1
      src_w_num = self.b + 1
      dst_x_num = 1
      dst_y_num = self.b + 1
      num_pools = nh
    src_x_step = 1
    src_y_step = 1 #self.W
    src_z_step = self.Th
    src_w_step = self.Tv #self.W * self.Tv
    dst_x_step = 1
    dst_y_step = 1 #self.W
    dst_z_num = 1
    dst_z_step = 1 #self.W * self.H
    for i in range(num_pools):
      (src_x_num, src_y_num, src_start, dst_start) = \
       self.ComputeEPVariables(edge_type, i)
      waveop = me_common_ds.WaveOpInfo(\
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
          , src_start\
          , dst_start\
          )
      if (edge_type == "UHEP"):
        self.pool_waveops.uhep_ops.append(waveop)
      elif (edge_type == "BHEP"):
        self.pool_waveops.bhep_ops.append(waveop)
      elif (edge_type == "LVEP"):
        self.pool_waveops.lvep_ops.append(waveop)
      else:
        self.pool_waveops.rvep_ops.append(waveop)
    return

  # CeP : Center Pool
  def ComputeCeP (self):
    src_x_num = self.S
    src_x_step = 1
    src_y_num = self.R
    src_y_step = 1# self.W
    src_z_num = self.a + 1
    src_z_step = self.Th
    src_w_num = self.b + 1
    src_w_step = self.Tv# self.W * self.Tv
    dst_x_num = self.a + 1
    dst_x_step = 1
    dst_y_num = self.b + 1
    dst_y_step = 1 #self.W
    dst_z_num = 1
    dst_z_step = 1 #self.W * self.H
    src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN\
        , math.ceil(self.pW / self.Th) * self.Th - self.pW)
    dst_start = (math.ceil(self.pN / self.Tv), math.ceil(self.pW / self.Th))
    waveop = me_common_ds.WaveOpInfo(\
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
        , src_start\
        , dst_start\
        )
    self.pool_waveops.cp_ops.append(waveop)
    return
  def ComputePool (self):
    self.ComputeCP("LTCP")
    self.ComputeCP("RTCP")
    self.ComputeCP("LBCP")
    self.ComputeCP("RBCP")
    self.ComputeEP("UHEP")
    self.ComputeEP("BHEP")
    self.ComputeEP("LVEP")
    self.ComputeEP("RVEP")
    self.ComputeCeP()
    self.ConstructPoolWaveOpInfo(self.pool_waveops.ltcp_ops\
        , self.pool_func + "_LTCP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.rtcp_ops\
        , self.pool_func + "_RTCP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.lbcp_ops\
        , self.pool_func + "_LBCP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.rbcp_ops\
        , self.pool_func + "_RBCP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.uhep_ops\
        , self.pool_func + "_UHEP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.bhep_ops\
        , self.pool_func + "_BHEP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.lvep_ops\
        , self.pool_func + "_LVEP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.rvep_ops\
        , self.pool_func + "_RVEP")
    self.ConstructPoolWaveOpInfo(self.pool_waveops.cp_ops\
        , self.pool_func + "_CP")

  def PrintWaves(self):
    print ("ltcp = []")
    for i in self.pool_waveops.ltcp_ops:
      i.print("ltcp")
    print ("rtcp = []")
    for i in self.pool_waveops.rtcp_ops:
      i.print("rtcp")
    print ("lbcp = []")
    for i in self.pool_waveops.lbcp_ops:
      i.print("lbcp")
    print ("rbcp = []")
    for i in self.pool_waveops.rbcp_ops:
      i.print("rbcp")
    print ("uhep = []")
    for i in self.pool_waveops.uhep_ops:
      i.print("uhep")
    print ("bhep = []")
    for i in self.pool_waveops.bhep_ops:
      i.print("bhep")
    print ("lvep = []")
    for i in self.pool_waveops.lvep_ops:
      i.print("lvep")
    print ("rvep = []")
    for i in self.pool_waveops.rvep_ops:
      i.print("rvep")
    print ("cp = []")
    for i in self.pool_waveops.cp_ops:
      i.print("cp")
    return

    # ops in WaveOpInfo defined in me_common_ds
  def ConstructPoolWaveOpInfo (self, ops, name_prefix):
      op_id = 0
      for i in ops:
          pool_frequency = i.src_z_num * i.src_w_num
          if (self.pool_func == "AvgPool"):
              pool_scale = 1 / pool_frequency
          else:
              pool_scale = 1
          if (self.ifmap.__class__.__name__ == "FMAPDim"):
              input_tensor = ""
          else:
              input_tensor = self.ifmap.waveop_name
          name = name_prefix + str(op_id)
          p = me_common_ds.PoolWaveOpInfo(\
              src_x_num = i.src_x_num\
              , src_x_step = i.src_x_step\
              , src_y_num = i.src_y_num\
              , src_y_step = i.src_y_step\
              , src_z_num = i.src_z_num\
              , src_z_step = i.src_z_step\
              , src_w_num = i.src_w_num\
              , src_w_step = i.src_w_step\
              , dst_x_num = i.dst_x_num\
              , dst_x_step = i.dst_x_step\
              , dst_y_num = i.dst_y_num\
              , dst_y_step = i.dst_y_step\
              , dst_z_num = i.dst_z_num\
              , dst_z_step = i.dst_z_step\
              , src_start = i.src_start\
              , dst_start = i.dst_start\
              , pool_frequency = pool_frequency\
              , pool_func = self.pool_func\
              , pool_scale = pool_scale\
              , prev_waveops = []\
              , input_tensor = input_tensor\
              , src_is_psum = self.src_is_psum\
              , dst_is_psum = self.dst_is_psum\
              , name = name\
              , ifmap = self.ifmap)
          self.waveops.append(p)
          op_id += 1
      return

  def ConvertWaveOpInfoToMEPoolSpec(self, waveopinfo):
    ifmap_lower =\
      me_utils.Coord(waveopinfo.src_start[1] + self.ifmap_tile_rect.lower.x\
                     , waveopinfo.src_start[0] + self.ifmap_tile_rect.lower.y)
    ifmap_upper =\
      ifmap_lower\
      + me_utils.Coord(waveopinfo.src_x_num\
        + (waveopinfo.src_z_num - 1) * waveopinfo.src_z_step - 1\
      , waveopinfo.src_y_num\
        +(waveopinfo.src_w_num - 1) * waveopinfo.src_w_step - 1)
    ifmap = me_utils.Rect(ifmap_lower, ifmap_upper)
#    print ("ifmap_lower = ", ifmap_lower)
#    print ("ifmap_upper = ", ifmap_upper)
#    print ("ifmap = ", ifmap)
    ofmap_lower =\
      me_utils.Coord(waveopinfo.dst_start[1]+self.ofmap_offset_x\
                     , waveopinfo.dst_start[0]+self.ofmap_offset_y)
    ofmap_upper =\
      ofmap_lower\
      + me_utils.Coord(waveopinfo.dst_x_num - 1\
      , waveopinfo.dst_y_num - 1)
    ofmap = me_utils.Rect(ofmap_lower, ofmap_upper)
    window_x = waveopinfo.src_x_num
    window_y = waveopinfo.src_y_num
    window = me_utils.Dim2D(window_x, window_y)
    stride = me_utils.Dim2D(waveopinfo.src_z_step, waveopinfo.src_w_step)
    return me_common_ds.MEPoolSpec(ifmap, ofmap, window, stride)

  def CollectMEPoolSpecs(self):
    pool_specs = []
    for w in self.waveops:
      pool_specs.append(self.ConvertWaveOpInfoToMEPoolSpec(w))
    return pool_specs
        

  def Decompose(self):
    result = []
#    if (self.padded_ifmap_tile_rect == self.ifmap_tile_rect):
#        result.append(me_common_ds.MEPoolSpec(self.ifmap_tile_rect\
#                  , self.ofmap_tile_rect\
#                  , self.pool_window_dim2d\
#                  , self.pool_stride_dim2d))
#    else:
    self.ComputePool()
    result = self.CollectMEPoolSpecs()
    return result

