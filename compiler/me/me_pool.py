import math

class FMAPDim:
  def __init__(self, H, W):
    self.H = H
    self.W = W
    return

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
    print(", src_start = ",self.src_start,"\\")
    print(", dst_start = ",self.dst_start,"\\")
    print(")")
    print(name+".append("+name+"_waveop)")
    return

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
  def __init__(self, ifmap, pool_window, padding):
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
    def getMinimum(a):
      int_a = int(a)
      return (int_a + 1)

    self.t = getMinimum((self.pN + self.H - self.R) / self.Tv)
    self.o = int((self.pN + self.pS + self.H - self.R) / self.Tv - self.t)
    self.u = getMinimum((self.pW + self.W - self.S) / self.Th)
    self.q = int((self.pW + self.W + self.pE - self.S) / self.Th - self.u)
    
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
      dst_start = (j, self.W - (self.q + 1) + i)
    elif (corner_type == "LBCP"):
      src_x_num = self.S - self.pW + i * self.Th
      src_y_num = self.H + self.pN - (self.t + j) * self.Tv
      src_start = (self.t * self.Tv - self.pN + j * self.Tv, 0)
      dst_start = (self.H - (self.o + 1) + j, i)
    else:
      src_x_num = self.W + self.pW - (self.u + i) * self.Th
      src_y_num = self.H + self.pN - (self.t + j) * self.Tv
      src_start = (self.t * self.Tv - self.pN + j * self.Tv\
          , self.u * self.Th - self.pW + i * self.Th)
      dst_start = (self.H - (self.o + 1) + j, self.W - (self.q + 1) + i)
    return (src_x_num, src_y_num, src_start, dst_start)

  def ComputeCP (self, corner_type):
    (nh, nv) = self.CP_ComputeHVMoveCount(corner_type)
    src_x_step = 1
    src_y_step = self.W
    src_z_num = 1
    src_z_step = self.Th
    src_w_num = 1
    src_w_step = self.W
    dst_x_num = 1
    dst_x_step = 1
    dst_y_num = 1
    dst_y_step = self.W
    for i in range(nh):
      for j in range(nv):
        (src_x_num, src_y_num, src_start, dst_start) = \
          self.ComputeCPVariables(corner_type, i, j)
        waveop = WaveOpInfo(\
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
        dst_start = (self.H - (self.o + 1) + i, math.ceil(self.pW / self.Th))
    else:
      src_y_num = self.R
      if (edge_type == "LVEP"):
        src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN, 0)
        dst_start = (math.ceil(self.pN / self.Tv), i)
        src_x_num = self.S - self.pW + i * self.Th
      else:
        src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN\
            , self.u * self.Th - self.pW + i * self.Th)
        dst_start = (math.ceil(self.pN / self.Tv), self.W - (self.q + 1) + i)
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
    src_y_step = self.W
    src_z_step = self.Th
    src_w_step = self.W * self.Tv
    dst_x_step = 1
    dst_y_step = self.W
    for i in range(num_pools):
      (src_x_num, src_y_num, src_start, dst_start) = \
       self.ComputeEPVariables(edge_type, i)
      waveop = WaveOpInfo(\
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
    src_y_step = self.W
    src_z_num = self.a + 1
    src_z_step = self.Th
    src_w_num = self.b + 1
    src_w_step = self.W * self.Tv
    dst_x_num = self.a + 1
    dst_x_step = 1
    dst_y_num = self.b + 1
    dst_y_step = self.W
    src_start = (math.ceil(self.pN / self.Tv) * self.Tv - self.pN\
        , math.ceil(self.pW / self.Th) * self.Th - self.pW)
    dst_start = (math.ceil(self.pN / self.Tv), math.ceil(self.pW / self.Th))
    waveop = WaveOpInfo(\
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
#    print ("ltcp = []")
#    for i in self.pool_waveops.ltcp_ops:
#      i.print("ltcp")
#    print ("rtcp = []")
#    for i in self.pool_waveops.rtcp_ops:
#      i.print("rtcp")
#    print ("lbcp = []")
#    for i in self.pool_waveops.lbcp_ops:
#      i.print("lbcp")
#    print ("rbcp = []")
#    for i in self.pool_waveops.rbcp_ops:
#      i.print("rbcp")
#    print ("uhep = []")
#    for i in self.pool_waveops.uhep_ops:
#      i.print("uhep")
#    print ("bhep = []")
#    for i in self.pool_waveops.bhep_ops:
#      i.print("bhep")
#    print ("lvep = []")
#    for i in self.pool_waveops.lvep_ops:
#      i.print("lvep")
#    print ("rvep = []")
#    for i in self.pool_waveops.rvep_ops:
#      i.print("rvep")
#    print ("cp = []")
#    for i in self.pool_waveops.cp_ops:
#      i.print("cp")
    return
