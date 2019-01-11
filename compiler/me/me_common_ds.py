"""
Common Data Structures in ME
"""
class FMAPDim:
  def __init__(self, H, W):
    self.H = H
    self.W = W
  
  def __str__(self):
      return "H = %d, W = %d"%(self.H, self.W)

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
    def __str__(self):
        return "N = %d, C = %d, H = %d, W = %d \
                file_name = %s waveop_name = %s"\
                %(self.N,self.C,self.H,self.W,self.file_name,self.waveop_name)


# Th : y direction stride
# Tv : x direction stride
class FilterSpec:
  def __init__(self, R, S, Th, Tv):
    self.R = R
    self.S = S
    self.Th = Th
    self.Tv = Tv
    return

class PaddingSpec:
  def __init__(self, pN, pS, pW, pE):
    self.pN = pN;
    self.pS = pS;
    self.pW = pW;
    self.pE = pE;
    return
  def __str__(self):
      return "pN = %d pS = %d pW = %d pE = %d"%(self.pN,self.pS,self.pW,self.pE)

class WaveOpInfo:
  def __init__(self):
      self.src_x_num = 0
      self.src_x_step = 0
      self.src_y_num = 0
      self.src_y_step = 0
      self.src_z_num = 0
      self.src_z_step = 0
      self.src_w_num = 0
      self.src_w_step = 0
      self.dst_x_num = 0
      self.dst_x_step = 0
      self.dst_y_num = 0
      self.dst_y_step = 0
      self.dst_z_num = 0
      self.dst_z_step = 0
      self.src_start = 0
      self.dst_start = 0
      return
  def __init__(
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
      ):
      self.src_x_num = src_x_num
      self.src_x_step = src_x_step
      self.src_y_num = src_y_num
      self.src_y_step = src_y_step
      self.src_z_num = src_z_num
      self.src_z_step = src_z_step
      self.src_w_num = src_w_num
      self.src_w_step = src_w_step
      self.dst_x_num = dst_x_num
      self.dst_x_step = dst_x_step
      self.dst_y_num = dst_y_num
      self.dst_y_step = dst_y_step
      self.dst_z_num = dst_z_num
      self.dst_z_step = dst_z_step
      self.src_start = src_start
      self.dst_start = dst_start
      return
  def __ne__(self, op):
    return ((self.src_x_num != op.src_x_num) |\
        (self.src_x_step != op.src_x_step) |\
        (self.src_y_num != op.src_y_num) |\
        (self.src_y_step != op.src_y_step) |\
        (self.src_z_num != op.src_z_num) |\
        (self.src_z_step != op.src_z_step) |\
        (self.src_w_num != op.src_w_num) |\
        (self.src_w_step != op.src_w_step) |\
        (self.dst_x_num != op.dst_x_num) |\
        (self.dst_x_step != op.dst_x_step) |\
        (self.dst_y_num != op.dst_y_num) |\
        (self.dst_y_step != op.dst_y_step) |\
        (self.src_start != op.src_start) |\
        (self.dst_start != op.dst_start))
  def __eq__(self, op):
    return ((self.src_x_num == op.src_x_num) &\
        (self.src_x_step == op.src_x_step) &\
        (self.src_y_num == op.src_y_num) &\
        (self.src_y_step == op.src_y_step) &\
        (self.src_z_num == op.src_z_num) &\
        (self.src_z_step == op.src_z_step) &\
        (self.src_w_num == op.src_w_num) &\
        (self.src_w_step == op.src_w_step) &\
        (self.dst_x_num == op.dst_x_num) &\
        (self.dst_x_step == op.dst_x_step) &\
        (self.dst_y_num == op.dst_y_num) &\
        (self.dst_y_step == op.dst_y_step) &\
        (self.src_start == op.src_start) &\
        (self.dst_start == op.dst_start))
  def print(self, name):
    print("%s = WaveOpInfo(%s"%(name+"_waveop","\\"))
    print("src_x_num = %d%s"%(self.src_x_num,"\\"))
    print(", src_x_step = %d%s"%(self.src_x_step, "\\"))
    print(", src_y_num = %d%s"%(self.src_y_num, "\\"))
    print(", src_y_step = %d%s"%(self.src_y_step, "\\"))
    print(", src_z_num = %d%s"%(self.src_z_num, "\\"))
    print(", src_z_step = %d%s"%(self.src_z_step, "\\"))
    print(", src_w_num = %d%s"%(self.src_w_num, "\\"))
    print(", src_w_step = %d%s"%(self.src_w_step, "\\"))
    print(", dst_x_num = %d%s"%(self.dst_x_num, "\\"))
    print(", dst_x_step = %d%s"%(self.dst_x_step, "\\"))
    print(", dst_y_num = %d%s"%(self.dst_y_num, "\\"))
    print(", dst_y_step = %d%s"%(self.dst_y_step, "\\"))
    print(", dst_z_num = %d%s"%(self.dst_z_num, "\\"))
    print(", dst_z_step = %d%s"%(self.dst_z_step, "\\"))
    print(", src_start = ",self.src_start,"\\")
    print(", dst_start = ",self.dst_start,"\\")
    print(")")
    print(name+".append("+name+"_waveop)")
    return

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

class MMWaveOpInfo(WaveOpInfo):
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
                 , start_tensor_calc\
                 , name\
                ):
        WaveOpInfo.__init__(\
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
        self.start_tensor_calc = start_tensor_calc
        self.name = name

    def print_prev_ops(self):
        for i in self.prev_waveops:
            print (i)

class PoolWaveOpInfo(WaveOpInfo):
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
                 , pool_func\
                 , pool_scale\
                 , prev_waveops\
                 , input_tensor\
                 , src_is_psum\
                 , dst_is_psum\
                 , name\
                 , ifmap\
                ):
        WaveOpInfo.__init__(\
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
        self.pool_func = pool_func
        self.pool_scale = pool_scale
        self.prev_waveops = prev_waveops
        self.input_tensor = input_tensor
        self.name = name
        self.src_is_psum = src_is_psum
        self.dst_is_psum = dst_is_psum
        self.ifmap = ifmap

    def print_prev_ops(self):
        for i in self.prev_waveops:
            print (i)

class MEPoolSpec:
    def __init__ (self, ifmap, ofmap, window, stride):
        self.ifmap = ifmap
        self.ofmap = ofmap
        self.window = window
        self.stride = stride
