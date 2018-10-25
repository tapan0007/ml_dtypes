# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Watchpoint insertion tool 
#
# Usage:
#
# python3 $KAENA_PATH/compiler/watchpoint/watchpoint_insert.py --kgraph compiler.json \
#   --wavegraph wavegraph.json -v --kgraph-node-list "activation_3/Relu activation_9/Relu" -v

import argparse
import logging
import json
import numpy as np
import re
import os
import sys

#
# Arch or Memory layout configurations
#
class Config:
  num_ofmap_channels = 64
  sb_partitions = 128  

#
# Errors and Exceptions
#

# exit code
ERR_WAVEGRAPH = 1
ERR_INTERNAL = 2

# Exception for invalid input wavegraph
class WavegraphException(Exception):
  pass

#
# Common utility functions
#
def _get_shape(format, shape):
  shape_dict = dict()
  for fmt, num in zip(format, shape):
    shape_dict[fmt] = num
  shape_dict['format'] = format
  return shape_dict


#
# State buffer memory region descriptor
#
class StateBufferRegion:
  # start, end: start and end addresses in a partition
  start = 0
  end = 0
  # p_start, p_end: start and end partition index.
  # p_start starts at 0 or 64.
  p_start = 0  
  p_end = 0

  def to_str(self):
    return '[' + str(self.start) + ', ' + str(self.end) + ') p[' + str(self.p_start) + ', ' + str(self.p_end) + ')'
  

#
# K-Graph node wrapper
#
class _KGraphNodeInfo:
  name = ''
  wnode_set = None
  ofmap_shape = None

  def __init__(self, name, ofmap_shape, ref_file):
    self.name = name
    self.wnode_set = []
    self.ofmap_shape = ofmap_shape
    self.ref_file = ref_file

  def get_wavegraph_node_set(self):
    return self.wnode_set

  @staticmethod
  def is_activation(layer_type):
    # TODO: Add more activation operators
    return layer_type == 'Relu' 
  
  @staticmethod
  def is_pool(layer_type):
    # TODO: Add more pool operators
    return layer_type == 'MaxPool' 

  @staticmethod
  def is_conv(layer_type):
    return layer_type == 'Conv'    

  @staticmethod
  def get_watchpoint_name_prefix(knode_name):
    return 'watchpoint_' + knode_name

  def get_ofmap_shape(self):
    return self.ofmap_shape

  def get_ref_file(self):
    return self.ref_file
    
#
# Wavegraph node wrapper
#
class _WaveGraphNodeInfo:
  wnode = None
  sb_region = None

  def __init__(self, wavegraph_node):
    self.wnode = wavegraph_node

  def get_sb_region(self):
    if self.sb_region == None:
      r = StateBufferRegion()
      r.start = _WaveGraphNodeInfo.compute_sb_start(self.wnode, False)
      r.end = r.start + _WaveGraphNodeInfo.compute_sb_access_size(self.wnode, False) 
      r.p_start, r.p_end = _WaveGraphNodeInfo.get_partition_range(self.wnode, False)
      self.sb_region = r
    return self.sb_region

  @staticmethod
  def get_out_dtype(wnode):
    return wnode['out_dtype']

  @staticmethod
  def get_type_bytes(dtype):
    if (dtype == 'bfloat16'): return 2
    else: return np.dtype(dtype).itemsize  

  def data_type_to_item_sz(data_type):
    return np.dtype(data_type).itemsize     

  @staticmethod
  def has_write_to_sb (op):
    op_type = op['waveop_type']
    if (op_type == 'SBAtomLoad'): 
      return True
    elif op_type == 'SBAtomSave':
      return False
    else:
      assert(('dst_is_psum' in op) and 'Expect dst_is_psum')
      return op['dst_is_psum'] == False       

  @classmethod
  def compute_access_size (cls, num, step, dtype):
      return (num * step) * cls.get_type_bytes(dtype)

  @classmethod
  def compute_access_size_w_access_pattern(cls, prefix, op):
      if (prefix == 'src'): type_prefix = 'in'
      else: type_prefix = 'out'

      z_num = prefix + '_z_num'
      y_num = prefix + '_y_num'
      x_num = prefix + '_x_num'
      z_step = prefix + '_z_step'
      y_step = prefix + '_y_step'
      x_step = prefix + '_x_step'
      dtype = type_prefix + '_dtype'
      if (op[z_num] == 1):
        if (op[y_num] == 1):
          size = cls.compute_access_size(
            op[x_num], op[x_step], op[dtype])
        else:
          size = cls.compute_access_size(
            op[y_num], op[y_step], op[dtype])
      else:
        size = cls.compute_access_size(
          op[z_num], op[z_step], op[dtype])
      return size

  @classmethod
  def compute_sb_access_size (cls, op, read):
    if (op['waveop_type'] == 'SBAtomSave' or op['waveop_type'] == 'SBAtomLoad'):
      size = op['length']
    else:
      if (read == True):
        size = cls.compute_access_size_w_access_pattern('src', op)
      elif (read == False):
        size = cls.compute_access_size_w_access_pattern('dst', op)
      else: size = None
    return size


  @staticmethod
  def compute_sb_start (op, read):
    if (op['waveop_type'] == 'SBAtomSave' or op['waveop_type'] == 'SBAtomLoad'):
      return op['sb_address']
    else:
      if (read == True):
        return op['src_sb_address']
      elif (read == False):
        return op['dst_sb_address']
      else:
        return None

  @staticmethod
  def get_partition_range(node, read):
    if 'start_at_mid_part' in node:
      mid_start = node['start_at_mid_part']
    elif ('src_start_at_mid_part' in node) or ('dst_start_at_mid_part' in node):
      mid_start = node['src_start_at_mid_part'] if read else node['dst_start_at_mid_part']
    else:
      raise ValueError("cannot find 'start_at_mid_part/src_start_at_mid_part/dst_start_at_mid_part' in " + node['waveop_name'])

    if 'num_partitions' in node:
      num_partitions = node['num_partitions']
    else:
      raise ValueError("cannot find 'num_partition' in " + node['waveop_name'])
    if not mid_start:
      return (0, num_partitions)
    else:
      return (Config.num_ofmap_channels, Config.num_ofmap_channels + num_partitions)

  @staticmethod
  def check_sb_overlap (region_a, region_b):
    overlap_h = (region_a.start >= region_b.start and region_a.start < region_b.end) or\
      (region_a.end > region_b.start and region_a.end <= region_b.end)
    overlap_v = (region_a.p_start >= region_b.p_start and region_a.p_start < region_b.p_end) or\
      (region_a.p_end > region_b.p_start and region_a.p_end <= region_b.p_end)
    return overlap_h and overlap_v

#
# Watchpoint insertion tool
#
class LowLevelWatchPointTool:

  # Input/Output graphs
  wavegraph = None
  kgraph = None
  wavegraph_out_json = ''

  # Analysis Info
  knode_infos = dict() # Set of _KGraphNodeInfo
  wnode_infos = dict() # Set of _WaveGraphNodeInfo
  
  # Misc. flags
  errors = 0
  verbose_enabled = False

  # wavegraph_json: input wavegraph json file
  # node_list: list of node names to watch
  def __init__(self, kgraph_json, wavegraph_json, verbose_enable = False):

    try:
      with open(kgraph_json) as f:
        self.kgraph = json.load(f)
    except EnvironmentError:
      raise WavegraphException("Cannot load " + kgraph_json)

    try:
      with open(wavegraph_json) as f:
        self.wavegraph = json.load(f)
    except EnvironmentError:
      raise WavegraphException('Cannot load ' + wavegraph_json)

    self.verbose_enabled = verbose_enable

    assert ("waveops" in self.wavegraph)

    root, ext = os.path.splitext(wavegraph_json)      
    self.wavegraph_out_json = root + '.watchpoint' + ext

    self._verbose('Graphs reading done')

  # Close wavegraph file
  def close(self):
    self._verbose('*** Writing wavegraph to ' + self.wavegraph_out_json)
    with (open(self.wavegraph_out_json, 'w')) as f:
        s = json.dumps(self.wavegraph, indent=2, sort_keys=True)
        s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
        s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
        f.write(s)    

  # Add multiple watchpoint nodes
  def insert_watchpoints(self, knode_list):   

    added_watchpoint_count = 0
    for knode_name in knode_list:

      matched_list = [n for n in self.kgraph['layers'] if n['layer_name'] == knode_name]
      if len(matched_list) != 1:
        raise WavegraphException('Cannot find k-graph node: ' + knode_name)

      knode = matched_list[0]

      # If we can get the following information without compiler.json or 'layers' in wavegraph.json,
      # we don't need to read k-graph.
      ofmap_shape = _get_shape(knode['ofmap_format'], knode['ofmap_shape'])
      ref_file = knode['ref_file']

      if ofmap_shape['format']  != 'NCHW':
        raise WavegraphException('Invalid ofmap shape: ' + tokens[0])

      added_watchpoint_count += self.insert_watchpoint(knode_name, ofmap_shape, ref_file)
    
    return added_watchpoint_count
 
  # Add a watchpoint node
  def insert_watchpoint(self, knode_name, ofmap_shape, ref_file):    
    knode_name = knode_name.strip()

    if not self._can_knode_have_watchpoint(knode_name, ofmap_shape, ref_file):
      raise WavegraphException('Cannot insert a watchpoint for ' + knode_name)

    return self._add_watchpoint_to_sb_dst_node(knode_name)

  def _verbose(self, msg):
    if self.verbose_enabled:
      print('INFO: ' + msg)

  def _warning(cls, msg):
    print('WARNING: ' + msg)

  def _create_kgraph_node_info(self, knode_name, ofmap_shape, ref_file):
 
    knode_info = _KGraphNodeInfo(knode_name, ofmap_shape, ref_file) 
    for wnode in self.wavegraph["waveops"]:
      if wnode['waveop_type'] == 'SBAtomLoad' or wnode['waveop_type'] == 'SBAtomSave':
        continue
      if wnode['layer_name'] == knode_name:
        # For Conv, we collect only pool wavegraph nodes generating the output of
        # the knode. For activation or pool, we assume that each wavegraph node
        # produces a portion of the knode output.
        if _WaveGraphNodeInfo.has_write_to_sb(wnode):
          knode_info.wnode_set.append(wnode) 

    if len(knode_info.wnode_set) == 0:      
      # We currently support only Activation and Pool
      wnode_type = knode_info.wnode_set[0]['waveop_type']
      if (wnode_type != 'Activation' and wnode_type != 'Pool'):
        raise WavegraphException('Only support activation or pool nodes')
      else:
        # Sanity check to check if all nodes have the same type.
        for wn in knode_info.wnode_set:
          if not (wn['waveop_type'] == wnode_type):
            raise WavegraphException('Inconsistent wavegraph node types: ' + wnode_type + ' and ' + wn['waveop_type'])

    self.knode_infos[knode_name] = knode_info        

    return self.knode_infos[knode_name]
  
  def _get_kgraph_node_info(self, knode_name):
    return self.knode_infos[knode_name]

  def _get_wavegraph_node_info(self, wnode):
    if not (wnode['waveop_name'] in self.wnode_infos):
      self.wnode_infos[wnode['waveop_name']] = _WaveGraphNodeInfo(wnode)
    return self.wnode_infos[wnode['waveop_name']]

  # Check if a wavenode can have a watchpoint node.
  # The current limitation is that knode must write to SB.
  def _can_knode_have_watchpoint(self, knode_name, ofmap_shape, ref_file):
    knode_info = self._create_kgraph_node_info(knode_name, ofmap_shape, ref_file)
    wnode_set = knode_info.get_wavegraph_node_set()
    # If empty, knode is a conv node without pool nodes.
    if not wnode_set:
      return False
    for wnode in wnode_set:
      if not ('dst_is_psum' in wnode):
        return False
      if wnode['dst_is_psum'] != False:
        return False
    return True

  # Insert a watchpoint for a kgraph node
  def _add_watchpoint_to_sb_dst_node(self, knode_name):

    watchpoint_node_name = _KGraphNodeInfo.get_watchpoint_name_prefix(knode_name)
    for wnode in self.wavegraph['waveops']:
      if wnode['waveop_name'].find(watchpoint_node_name) == 0:
        self._warning('There already exists a watchpoint for ' + knode_name)
        return 0

    knode_info = self._get_kgraph_node_info(knode_name)

    # wavegraph nodes corresponding to knode
    wnode_set = knode_info.get_wavegraph_node_set() 
    if not wnode_set:
      raise WavegraphException('No wavegraph nodes writing to SB are found in ' + knode_name)

    self._verbose('*** Adding watchpoint to knode ' + knode_name)
    self._verbose('Wavegraph nodes:')

    # Calculate the SB region of knode by combining distinct w-node SB regions.
    # We may want to check if all w-node SB regions are distinct.
    knode_sb_lb = sys.maxsize
    knode_sb_ub = 0
    knode_sb_p_start = sys.maxsize
    knode_sb_p_end = 0 
    knode_shape = knode_info.get_ofmap_shape()
    wnode_total_bytes = 0
    for wnode in wnode_set:
      wnode_info = self._get_wavegraph_node_info(wnode)
      wnode_sb_region = wnode_info.get_sb_region()
      knode_sb_lb = min(knode_sb_lb, wnode_sb_region.start)
      knode_sb_ub = max(knode_sb_ub, wnode_sb_region.end)   
      knode_sb_p_start = min(knode_sb_p_start, wnode_sb_region.p_start)  
      knode_sb_p_end = max(knode_sb_p_end, wnode_sb_region.p_end)
      wnode_total_bytes += (wnode_sb_region.end - wnode_sb_region.start) * (wnode_sb_region.p_end - wnode_sb_region.p_start)
      
      self._verbose('  Wave node:' + wnode['waveop_name'] + ': ' + wnode_sb_region.to_str())
    
    self._verbose('K-Graph SB region: {0}-{1} p{2}-{3}'.format(knode_sb_lb, knode_sb_ub, knode_sb_p_start, knode_sb_p_end))
 
    # Cacluate k-node size
    data_type = _WaveGraphNodeInfo.get_out_dtype(wnode_set[0])
    data_type_bytes = _WaveGraphNodeInfo.get_type_bytes(data_type)
    knode_shape_bytes =  data_type_bytes * knode_shape['N'] * knode_shape['C'] * knode_shape['H'] * knode_shape['W']
    
    # Check if wnodes cover the entire output of knode. If we are interested
    # in a portition of k-node output, we don't want to check this.
    assert(knode_sb_p_start == 0 and wnode_total_bytes == knode_shape_bytes)

    # watchpoint file names
    root, ext = os.path.splitext(knode_info.get_ref_file())
    ref_file = root + '-simout' + ext
    #ref_file = 'watchpoint_' + knode_name.replace('/','_').replace('\\','_') + '.npy'
    
    # W * H
    image_wh_bytes = knode_shape['H'] * knode_shape['W'] * data_type_bytes
    
    # DRAM step for images (W * H)    
    dram_partition_steps = image_wh_bytes

    # Find insertion point and insert save_node
    insert_idx = -1
    for idx, node in enumerate(self.wavegraph['waveops']):
      if node in wnode_set:
        insert_idx = idx
    assert(insert_idx != -1)
    
    # Create save nodes
    #
    # - ofmap is divided (folded) into c-tiles if the number of channel is >= 128
    # - each c-tile saves ofmap WxH planes for up-to 128 channels.

    last_save_node = None
    num_n_tiles = knode_shape['N']
    num_c_tiles = int((knode_shape['C'] + Config.sb_partitions - 1) / Config.sb_partitions)    
    sb_c_stride_bytes = image_wh_bytes     
    sb_n_stride_bytes = sb_c_stride_bytes * num_c_tiles    
    save_node_name_prefix = _KGraphNodeInfo.get_watchpoint_name_prefix(knode_name)
    num_save_nodes = 0

    for n in range(0, num_n_tiles):

      sb_n_start = knode_sb_lb + n * sb_n_stride_bytes
      remain_ofmap_channels = knode_shape['C']
      file_m_offset = n * dram_partition_steps * knode_shape['C']

      for c in range(0, num_c_tiles):

        file_c_offset = file_m_offset + c * dram_partition_steps * min(128, knode_shape['C'])
        sb_c_start = sb_n_start + c * sb_c_stride_bytes
        save_num_partitions = min(Config.sb_partitions, remain_ofmap_channels)
        remain_ofmap_channels -= Config.sb_partitions

        if last_save_node == None:
          save_predecessors = [ node['waveop_name'] for node in wnode_set ]
        else:
          save_predecessors = [ last_save_node['waveop_name'] ]

        last_save_node =  {
            'previous_waveops'    : save_predecessors,
            'waveop_type'         : 'SBAtomSave',
            'waveop_name'         : save_node_name_prefix + '_n' + str(n) + '_c' + str(c),
            'layer_name'          : knode_name,
            'sb_address'          : sb_c_start,
            'data_type'           : data_type,
            'ref_file'            : ref_file,
            'ref_file_sz'         : knode_shape_bytes,
            'ref_file_format'     : knode_shape['format'],
            'ref_file_shape'      : [ knode_shape['N'], knode_shape['C'], knode_shape['H'], knode_shape['W'] ],
            'offset_in_file'      : file_c_offset,
            'length'              : sb_c_stride_bytes,
            'start_at_mid_part'   : False, 
            'ofmaps_fold_idx'     : 0,   # TODO: is this still needed?
            'batch_fold_idx'      : 0,   # TODO: is this still needed?
            'num_partitions'      : save_num_partitions, # ofmap_num_channels or folded by 128-partition chunks.
            'partition_step_bytes': dram_partition_steps, 
            'last_save_of_file'   : False,
            'final_layer_ofmap'   : False,
          }

        self.wavegraph['waveops'].insert(insert_idx+1, last_save_node)  

        self._verbose('Adding watchpoint save:' + last_save_node['waveop_name'])
        
        insert_idx += 1
        num_save_nodes += 1

    # Find nodes that are WAR-dependent on each wave node and add edges from save node
    # to those nodes. Assuming that SB regions from wave nodes are distinct, it iswatch
    # equivalent to find all WAR-dependent nodes of the original K-Graph node.
    sb_waw_dep_nodes = []
    for wnode in wnode_set:
      sb_waw_dep_nodes += self._find_sb_war_dependent_node(wnode)
    
    for node in sb_waw_dep_nodes:
      node["previous_waveops"].append(last_save_node['waveop_name'])

    self._verbose('Created watchpoint saves: node_refix: {0}: ref_file: {1}'.format(save_node_name_prefix, ref_file))

    return num_save_nodes

  # Find nodes such that their SB regions overlap with the SB region of wnode.
  def _find_sb_war_dependent_node(self, wnode):

    sb_dep_nodes = []
    wnode_list = self.wavegraph["waveops"]
    
    # Assume that waveops list is topologically sorted and
    # so dependents nodes should be below wnode.
    for wnode_idx, node in enumerate(wnode_list):
      if node["waveop_name"] == wnode["waveop_name"]:
        break
        
    assert(wnode_idx < len(wnode_list))
    assert(_WaveGraphNodeInfo.has_write_to_sb(wnode))

    wnode_info = self._get_wavegraph_node_info(wnode)
    wnode_sb_region = wnode_info.get_sb_region()
    
    self._verbose('WAR dependent nodes for ' + wnode['waveop_name'] + 
      ' with SB region ' + wnode_sb_region.to_str() + ':')
    
    # find wavegraph nodes dependent on wnode.
    for succ in wnode_list[wnode_idx+1:]:
      if not _WaveGraphNodeInfo.has_write_to_sb(succ) :
        continue
      succ_info = self._get_wavegraph_node_info(succ)
      succ_sb_region = succ_info.get_sb_region()        
      if _WaveGraphNodeInfo.check_sb_overlap(wnode_sb_region, succ_sb_region) :
        sb_dep_nodes.append(succ)
        self._verbose('  ' + succ['waveop_name'] + ': addr ' + succ_sb_region.to_str())

    return sb_dep_nodes



# main function
def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("--kgraph", default="compiler.json", help="K-graph Json file to write; defaults to compiler.json")
  parser.add_argument("--wavegraph", default="wavegraph.json", help="Wave-graph Json file to write; defaults to wavegraph.json")
  parser.add_argument('--kgraph-node-list', default=[], nargs='+')
  parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print out verbose messages')
  args = parser.parse_args() 

  try:
    llpt = LowLevelWatchPointTool(args.kgraph, args.wavegraph, args.verbose)
    llpt.insert_watchpoints(args.kgraph_node_list)
    llpt.close()
  except WavegraphException as e:
    print(e.args[0]) 
    return ERR_WAVEGRAPH
  except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    return ERR_INTERNAL
  
  return 0

if __name__ == "__main__":
  sys.exit(main())
 