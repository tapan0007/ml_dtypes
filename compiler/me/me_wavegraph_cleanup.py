import json
import re
import numpy as np

waveops = dict()
weight_waveops = dict() # key : start address

def update_wavegraph(layer, predecessors, closest_waveop):
    prev_waveops = layer["previous_waveops"]
    for p in prev_waveops:
        if p in predecessors:
            prev_waveops.remove(p)
            if (closest_waveop["waveop_name"] not in prev_waveops):
                prev_waveops.append(closest_waveop["waveop_name"])
        else:
            predecessors.add(p)

def compute_sb_start (op, read):
  if (op['waveop_type'] == 'SBAtomSave' or op['waveop_type'] == 'SBAtomLoad'):
    return op['sb_address']
  else:
    if (read == True):
#      print ("op = %s"%op['waveop_name'])
      return op['src_sb_address']
    elif (read == False):
      return op['dst_sb_address']
    else:
      return None

def compute_start_pos (start, num, step, dtype):
    return start + (num - 1) * step * np.dtype(dtype).itemsize

def compute_access_size_w_access_pattern(prefix, op):
    if (prefix == 'src'): type_prefix = 'in'
    else: type_prefix = 'out'

    w_num = prefix + '_w_num'
    z_num = prefix + '_z_num'
    y_num = prefix + '_y_num'
    x_num = prefix + '_x_num'
    w_step = prefix + '_w_step'
    z_step = prefix + '_z_step'
    y_step = prefix + '_y_step'
    x_step = prefix + '_x_step'
    dtype = type_prefix + '_dtype'

    w_start = 0
    z_start = 0
    y_start = 0
    x_start = 0
    if (w_num in op):
        w_start = compute_start_pos (0, op[w_num], op[w_step], op[dtype])
    if (z_num in op):
        z_start = compute_start_pos (w_start, op[z_num], op[z_step], op[dtype])
    if (y_num in op):
        y_start = compute_start_pos (z_start, op[y_num], op[y_step], op[dtype])
    if (x_num in op):
        x_start = compute_start_pos (y_start, op[x_num], op[x_step], op[dtype])
    return (x_start + np.dtype(op[dtype]).itemsize - 1)

def compute_sb_access_size (op, read):
  if (op['waveop_type'] == 'SBAtomSave' or op['waveop_type'] == 'SBAtomLoad'):
    size = op['length']
  else:
    if (read == True):
      size = compute_access_size_w_access_pattern('src', op)
    elif (read == False):
      size = compute_access_size_w_access_pattern('dst', op)
    else: size = None
  return size

def determine_read_write (op):
  read = True
  if (op['waveop_type'] == 'SBAtomLoad'): read = False
  elif (op['waveop_type'] == 'Activation' or op['waveop_type'] == 'Pool'):
    if (op['src_is_psum'] == True and op['dst_is_psum'] == False):
      read = False
    elif (op['src_is_psum'] == True and op['dst_is_psum'] == True):
      read = None
  return read

def check_overlap (a_start, a_end, b_start, b_end):
    conf = False
    if ((a_start <= b_end and a_end >= b_end) or\
        (a_start <= b_start and a_end>=b_start) or\
        (a_start >= b_start and a_end <= b_end)):
      conf = True
    return conf

def conflict (cur_op, prev_op):
  prev_read = determine_read_write(prev_op)
  prev_sb_start = compute_sb_start(prev_op, prev_read)
  prev_acc_size = compute_sb_access_size(prev_op, prev_read) 
  prev_sb_end = prev_sb_start + prev_acc_size - 1
  cur_read = determine_read_write(cur_op)
  cur_sb_start = compute_sb_start(cur_op, cur_read)
  cur_acc_size = compute_sb_access_size(cur_op, cur_read) 
  cur_sb_end = cur_sb_start + cur_acc_size - 1
  conf = check_overlap(cur_sb_start,cur_sb_end,prev_sb_start,prev_sb_end)
  if (conf == False):
    print("cur_op = %s"%cur_op['waveop_name'])
    print("prev_op = %s"%prev_op['waveop_name'])
    print ("1.prev_start = %d, prev_end = %d"%(prev_sb_start, prev_sb_end))
    if (prev_op['waveop_type'] == 'MatMul'):
      if (prev_op['weights_sb_address'] != -1):
        prev_sb_start = prev_op['weights_sb_address']
        num_weight_elem = prev_op['num_column_partitions']
        prev_sb_end=\
          prev_sb_start+num_weight_elem*np.dtype(prev_op['in_dtype']).itemsize
        conf = check_overlap(cur_sb_start,cur_sb_end,prev_sb_start,prev_sb_end)
    if (prev_op['waveop_type'] == 'Activation' and conf == False):
      print ("2.prev_start = %d, prev_end = %d"%(prev_sb_start, prev_sb_end))
      if (prev_op['bias_add_en'] == True):
        prev_sb_start = prev_op['bias_sb_address']
        prev_sb_end = prev_sb_start + np.dtype(prev_op['bias_dtype']).itemsize - 1
        conf = check_overlap(cur_sb_start,cur_sb_end,prev_sb_start,prev_sb_end)
      if (prev_op['dst_is_psum'] == False and conf == False):
        print ("3.prev_start = %d, prev_end = %d"%(prev_sb_start, prev_sb_end))
        prev_sb_start = prev_op['dst_sb_address']
        prev_acc_size = compute_sb_access_size(prev_op, False) 
        prev_sb_end = prev_sb_start + prev_acc_size - 1
        conf = check_overlap(cur_sb_start,cur_sb_end,prev_sb_start,prev_sb_end)
    if (prev_op['waveop_type'] == 'Pool' and conf == False):
      print ("4.prev_start = %d, prev_end = %d"%(prev_sb_start, prev_sb_end))
      if (prev_op['dst_is_psum'] == False):
        prev_sb_start = prev_op['dst_sb_address']
        prev_acc_size = compute_sb_access_size(prev_op, False) 
        prev_sb_end = prev_sb_start + prev_acc_size - 1
        conf = check_overlap(cur_sb_start,cur_sb_end,prev_sb_start,prev_sb_end)
    if (conf == False):
      print ("5.prev_start = %d, prev_end = %d"%(prev_sb_start, prev_sb_end))
      print ("cur_start = %d, cur_end = %d"%(cur_sb_start, cur_sb_end))
  return conf

def remove_wrong_data_dependency (wavegraph):
    if ("waveops" in wavegraph):
        layers = wavegraph["waveops"]
        for l in layers:
            waveops[l['waveop_name']] = l
            waveop_type = l['waveop_type']
            if (waveop_type == "SBAtomLoad"):
                for p in l['previous_waveops']:
                  if (conflict(l, waveops[p]) == False):
                    l['previous_waveops'].remove(p)
                    print ("%s is removed from previous_waveops of %s"%(
                          p, l['waveop_name']))

def remove_redundant_edges (wavegraph_json, is_file = False):
    total_engines = 4
    pe = 0
    act = 1
    pool = 2
    dma = 3

    predecessors = []
    closest_waveop = []
#    waveops = dict()
    for i in range(total_engines):
        predecessors.append(set())
        closest_waveop.append("")
    if (is_file == True):
        with open(wavegraph_json) as f:
            wavegraph = json.load(f)
    else:
        wavegraph = wavegraph_json
    remove_wrong_data_dependency(wavegraph)
    if ("waveops" in wavegraph):
        layers = wavegraph["waveops"]
        for l in layers:
#            waveops[l['waveop_name']] = l
            waveop_type = l["waveop_type"]
            update = True
            if (waveop_type == "MatMul"):
                predecessor = predecessors[pe]
                wop = closest_waveop[pe]
                closest_waveop[pe] = l
            elif (waveop_type == "Pool" or waveop_type == "ResAdd"):
                predecessor = predecessors[pool]
                wop = closest_waveop[pool]
                closest_waveop[pool] = l
            elif (waveop_type == "Activation"):
                predecessor = predecessors[act]
                wop = closest_waveop[act]
                closest_waveop[act] = l
#            elif (waveop_type == "SBAtomLoad" or waveop_type == "SBAtomSave"):
            elif (waveop_type == "SBAtomLoad"):
                predecessor = predecessors[dma]
                wop = closest_waveop[dma]
                closest_waveop[dma] = l
                update = False
            elif (waveop_type == "SBAtomSave"):
                predecessor = predecessors[dma]
                wop = closest_waveop[dma]
                closest_waveop[dma] = l
                update = False
            else: 
                print (waveop_type)
                assert(0)
            if (update == True):
              update_wavegraph(l, predecessor, wop)
        if (is_file == True):
            o_file_name = re.sub(r'\.json','',wavegraph_json)
            o_file_name += "_new.json"
            with open(o_file_name, 'w') as outfile:
                s = json.dumps(wavegraph, indent=2, sort_keys = True)
                s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
                s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
                outfile.write(s)
    return wavegraph

#remove_redundant_edges("wavegraph.json-b", True)
