import unittest

import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_pool

#ifmap = me_pool.FMAPDim(6, 6)
#pool_window = me_pool.FilterSpec(3, 3, 2, 2)
#padding = me_pool.PaddingSpec(3, 4, 3, 4)
#pool = me_pool.Pool(ifmap, pool_window, padding)
#pool.ComputePool()

class TestPoolDecomposition(unittest.TestCase):
  def compare_geometry (self, golden, test):
    if (golden != test):
      return False
    else:
      return True
    return

  def _4x4ifmap_3x3window_1_1_stride(self):
    print ("INFO::test_4x4ifmap_3x3window_1_1_stride")
    ltcp = []
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (0, 0) \
        )
    ltcp.append(ltcp_waveop)
    rtcp = []
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 2) \
        , dst_start =  (0, 3) \
        )
    rtcp.append(rtcp_waveop)
    lbcp = []
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 0) \
        , dst_start =  (3, 0) \
        )
    lbcp.append(lbcp_waveop)
    rbcp = []
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 2) \
        , dst_start =  (3, 3) \
        )
    rbcp.append(rbcp_waveop)
    uhep = []
    uhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 2\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (0, 1) \
        )
    uhep.append(uhep_waveop)
    bhep = []
    bhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 2\
        , src_z_step = 1\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 0) \
        , dst_start =  (3, 1) \
        )
    bhep.append(bhep_waveop)
    lvep = []
    lvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 2\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (1, 0) \
        )
    lvep.append(lvep_waveop)
    rvep = []
    rvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 1\
        , src_w_num = 2\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 2) \
        , dst_start =  (1, 3) \
        )
    rvep.append(rvep_waveop)
    cp = []
    cp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 2\
        , src_z_step = 1\
        , src_w_num = 2\
        , src_w_step = 4\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (1, 1) \
        )
    cp.append(cp_waveop)
    ifmap = me_pool.FMAPDim(4, 4)
    pool_window = me_pool.FilterSpec(3, 3, 1, 1)
    padding = me_pool.PaddingSpec(1, 1, 1, 1)
    pool = me_pool.Pool(ifmap, pool_window, padding)
    pool.ComputePool()
    for i in range(len(pool.pool_waveops.ltcp_ops)):
      if (self.compare_geometry(ltcp[i], pool.pool_waveops.ltcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rtcp_ops)):
      if (self.compare_geometry(rtcp[i], pool.pool_waveops.rtcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lbcp_ops)):
      if (self.compare_geometry(lbcp[i], pool.pool_waveops.lbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rbcp_ops)):
      if (self.compare_geometry(rbcp[i], pool.pool_waveops.rbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.uhep_ops)):
      if (self.compare_geometry(uhep[i], pool.pool_waveops.uhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.bhep_ops)):
      if (self.compare_geometry(bhep[i], pool.pool_waveops.bhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rvep_ops)):
      if (self.compare_geometry(rvep[i], pool.pool_waveops.rvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lvep_ops)):
      if (self.compare_geometry(lvep[i], pool.pool_waveops.lvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.cp_ops)):
      if (self.compare_geometry(cp[i], pool.pool_waveops.cp_ops[i]) ==\
          False):
        return False
    return True

  def test_4x4ifmap_3x3window_1_1_stride(self):
    self.assertTrue(self._4x4ifmap_3x3window_1_1_stride())

  def _4x4ifmap_3x3window_2_2_stride(self):
    print ("INFO::test_4x4ifmap_3x3window_2_2_stride")
    ltcp = []
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (0, 0) \
        )
    ltcp.append(ltcp_waveop)
    rtcp = []
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 2) \
        , dst_start =  (0, 2) \
        )
    rtcp.append(rtcp_waveop)
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 4) \
        , dst_start =  (0, 3) \
        )
    rtcp.append(rtcp_waveop)
    lbcp = []
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 0) \
        , dst_start =  (2, 0) \
        )
    lbcp.append(lbcp_waveop)
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (4, 0) \
        , dst_start =  (3, 0) \
        )
    lbcp.append(lbcp_waveop)
    rbcp = []
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 2) \
        , dst_start =  (2, 2) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (4, 2) \
        , dst_start =  (3, 2) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 4) \
        , dst_start =  (2, 3) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 4\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (4, 4) \
        , dst_start =  (3, 3) \
        )
    rbcp.append(rbcp_waveop)
    uhep = []
    uhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (0, 1) \
        )
    uhep.append(uhep_waveop)
    bhep = []
    bhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (2, 0) \
        , dst_start =  (2, 1) \
        )
    bhep.append(bhep_waveop)
    bhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (4, 0) \
        , dst_start =  (3, 1) \
        )
    bhep.append(bhep_waveop)
    lvep = []
    lvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (1, 0) \
        )
    lvep.append(lvep_waveop)
    rvep = []
    rvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 2) \
        , dst_start =  (1, 2) \
        )
    rvep.append(rvep_waveop)
    rvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 4) \
        , dst_start =  (1, 3) \
        )
    rvep.append(rvep_waveop)
    cp = []
    cp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 4\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 8\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 4\
        , dst_z_num = 1\
        , dst_z_step = 16\
        , src_start =  (0, 0) \
        , dst_start =  (1, 1) \
        )
    cp.append(cp_waveop)
    ifmap = me_pool.FMAPDim(4, 4)
    pool_window = me_pool.FilterSpec(3, 3, 2, 2)
    padding = me_pool.PaddingSpec(2, 3, 2, 3)
    pool = me_pool.Pool(ifmap, pool_window, padding)
    pool.ComputePool()
    for i in range(len(pool.pool_waveops.ltcp_ops)):
      if (self.compare_geometry(ltcp[i], pool.pool_waveops.ltcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rtcp_ops)):
      if (self.compare_geometry(rtcp[i], pool.pool_waveops.rtcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lbcp_ops)):
      if (self.compare_geometry(lbcp[i], pool.pool_waveops.lbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rbcp_ops)):
      if (self.compare_geometry(rbcp[i], pool.pool_waveops.rbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.uhep_ops)):
      if (self.compare_geometry(uhep[i], pool.pool_waveops.uhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.bhep_ops)):
      if (self.compare_geometry(bhep[i], pool.pool_waveops.bhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rvep_ops)):
      if (self.compare_geometry(rvep[i], pool.pool_waveops.rvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lvep_ops)):
      if (self.compare_geometry(lvep[i], pool.pool_waveops.lvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.cp_ops)):
      if (self.compare_geometry(cp[i], pool.pool_waveops.cp_ops[i]) ==\
          False):
        return False
    return True

  def test_4x4ifmap_3x3window_2_2_stride(self):
    self.assertTrue(self._4x4ifmap_3x3window_2_2_stride())
    return

  def _6x6ifmap_3x3window_2_2_stride(self):
    print ("INFO::test_6x6ifmap_3x3window_2_2_stride")
    ltcp = []
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 0) \
        , dst_start =  (0, 0) \
        )
    ltcp.append(ltcp_waveop)
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 0) \
        , dst_start =  (1, 0) \
        )
    ltcp.append(ltcp_waveop)
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 0) \
        , dst_start =  (0, 1) \
        )
    ltcp.append(ltcp_waveop)
    ltcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 0) \
        , dst_start =  (1, 1) \
        )
    ltcp.append(ltcp_waveop)
    rtcp = []
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 5) \
        , dst_start =  (0, 4) \
        )
    rtcp.append(rtcp_waveop)
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 5) \
        , dst_start =  (1, 4) \
        )
    rtcp.append(rtcp_waveop)
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = -1\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 7) \
        , dst_start =  (0, 5) \
        )
    rtcp.append(rtcp_waveop)
    rtcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = -1\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 7) \
        , dst_start =  (1, 5) \
        )
    rtcp.append(rtcp_waveop)
    lbcp = []
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (5, 0) \
        , dst_start =  (4, 0) \
        )
    lbcp.append(lbcp_waveop)
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = -1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (7, 0) \
        , dst_start =  (5, 0) \
        )
    lbcp.append(lbcp_waveop)
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (5, 0) \
        , dst_start =  (4, 1) \
        )
    lbcp.append(lbcp_waveop)
    lbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = -1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (7, 0) \
        , dst_start =  (5, 1) \
        )
    lbcp.append(lbcp_waveop)
    rbcp = []
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (5, 5) \
        , dst_start =  (4, 4) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = -1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (7, 5) \
        , dst_start =  (5, 4) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = -1\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (5, 7) \
        , dst_start =  (4, 5) \
        )
    rbcp.append(rbcp_waveop)
    rbcp_waveop = me_pool.WaveOpInfo(\
        src_x_num = -1\
        , src_x_step = 1\
        , src_y_num = -1\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 6\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (7, 7) \
        , dst_start =  (5, 5) \
        )
    rbcp.append(rbcp_waveop)
    uhep = []
    uhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 0\
        , src_y_step = 6\
        , src_z_num = 2\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 12\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 1) \
        , dst_start =  (0, 2) \
        )
    uhep.append(uhep_waveop)
    uhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 2\
        , src_y_step = 6\
        , src_z_num = 2\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 12\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (0, 1) \
        , dst_start =  (1, 2) \
        )
    uhep.append(uhep_waveop)
    bhep = []
    bhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 1\
        , src_y_step = 6\
        , src_z_num = 2\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 12\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (5, 1) \
        , dst_start =  (4, 2) \
        )
    bhep.append(bhep_waveop)
    bhep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = -1\
        , src_y_step = 6\
        , src_z_num = 2\
        , src_z_step = 2\
        , src_w_num = 1\
        , src_w_step = 12\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 1\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (7, 1) \
        , dst_start =  (5, 2) \
        )
    bhep.append(bhep_waveop)
    lvep = []
    lvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 0\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 2\
        , src_w_step = 12\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (1, 0) \
        , dst_start =  (2, 0) \
        )
    lvep.append(lvep_waveop)
    lvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 2\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 2\
        , src_w_step = 12\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (1, 0) \
        , dst_start =  (2, 1) \
        )
    lvep.append(lvep_waveop)
    rvep = []
    rvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = 1\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 2\
        , src_w_step = 12\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (1, 5) \
        , dst_start =  (2, 4) \
        )
    rvep.append(rvep_waveop)
    rvep_waveop = me_pool.WaveOpInfo(\
        src_x_num = -1\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 6\
        , src_z_num = 1\
        , src_z_step = 2\
        , src_w_num = 2\
        , src_w_step = 12\
        , dst_x_num = 1\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (1, 7) \
        , dst_start =  (2, 5) \
        )
    rvep.append(rvep_waveop)
    cp = []
    cp_waveop = me_pool.WaveOpInfo(\
        src_x_num = 3\
        , src_x_step = 1\
        , src_y_num = 3\
        , src_y_step = 6\
        , src_z_num = 2\
        , src_z_step = 2\
        , src_w_num = 2\
        , src_w_step = 12\
        , dst_x_num = 2\
        , dst_x_step = 1\
        , dst_y_num = 2\
        , dst_y_step = 6\
        , dst_z_num = 1\
        , dst_z_step = 36\
        , src_start =  (1, 1) \
        , dst_start =  (2, 2) \
        )
    cp.append(cp_waveop)
    ifmap = me_pool.FMAPDim(6, 6)
    pool_window = me_pool.FilterSpec(3, 3, 2, 2)
    padding = me_pool.PaddingSpec(3, 4, 3, 4)
    pool = me_pool.Pool(ifmap, pool_window, padding)
    pool.ComputePool()
    for i in range(len(pool.pool_waveops.ltcp_ops)):
      if (self.compare_geometry(ltcp[i], pool.pool_waveops.ltcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rtcp_ops)):
      if (self.compare_geometry(rtcp[i], pool.pool_waveops.rtcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lbcp_ops)):
      if (self.compare_geometry(lbcp[i], pool.pool_waveops.lbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rbcp_ops)):
      if (self.compare_geometry(rbcp[i], pool.pool_waveops.rbcp_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.uhep_ops)):
      if (self.compare_geometry(uhep[i], pool.pool_waveops.uhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.bhep_ops)):
      if (self.compare_geometry(bhep[i], pool.pool_waveops.bhep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.rvep_ops)):
      if (self.compare_geometry(rvep[i], pool.pool_waveops.rvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.lvep_ops)):
      if (self.compare_geometry(lvep[i], pool.pool_waveops.lvep_ops[i]) ==\
          False):
        return False
    for i in range(len(pool.pool_waveops.cp_ops)):
      if (self.compare_geometry(cp[i], pool.pool_waveops.cp_ops[i]) ==\
          False):
        return False
    return True

  def test_6x6ifmap_3x3window_2_2_stride(self):
    self.assertTrue(self._6x6ifmap_3x3window_2_2_stride())


if __name__ == '__main__':
    unittest.main()
