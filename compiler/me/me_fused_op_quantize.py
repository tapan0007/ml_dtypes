"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Fused operations execution for quantized operators, including verification and waveop generation
"""
from me_fused_op import *


# Execute a dequantize operator
def execute_dequantize_op(self, tpb, batch_item):
    first_op = self[0]
    dequant_scale = first_op.data['dequant_scale']
    zero_point = first_op.data['zero_point']
    first_op.ofmaps_file_params.dram_data = tpb.pool.dequantize(first_op.ifmaps_file_params.dram_data, dequant_scale, zero_point)
    n_id = batch_item // first_op.Tn
    for m_id in range(first_op.m):
        for h_id in range(first_op.h):
            for w_id in range(first_op.w):
                tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False)
                ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = self.first_op.dst_is_psum)
                execute_dequantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                emit_waveops_dequantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                # execute subsequent instructions
                if len(self) > 1:
                    self.execute_postconv_tile_ops(tpb, ofmap_tile, ofmap_tile, psum_bank_id)

def execute_dequantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
    first_op = self.first_op
    ifmap_subtile = ifmap_tile.make_pewave()
    ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
    input_tilex = ifmap_tile.tile_rect.dim2d.x
    input_tiley = ifmap_tile.tile_rect.dim2d.y
    ifmaps_data_extract = ifmaps_data[0:input_tiley*input_tilex*first_op.Tn, :]
    layer_type = first_op.data['layer_type']
    assert layer_type == 'Dequantize'
    dequant_scale = self.first_op.data['dequant_scale']
    zero_point = self.first_op.data['zero_point']
    tile_data_flatten = tpb.pool.dequantize(ifmaps_data_extract, dequant_scale, zero_point)
    if first_op.dst_is_psum:
        tpb.pearray.write_psum(psum_bank_id, 0, tile_data_flatten.shape[0], tile_data_flatten)

def emit_waveops_dequantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
    # wave loop ordering scheme: nmhw
    first_op = self.first_op
    ifmap_tile_subtile \
            = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, Dim2D(1,1))
    ofmap_tile_subtile \
            = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, Dim2D(1,1))
    layer_type = first_op.data['layer_type']
    reader_engine = EngineEnum.POOL
    (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) \
        = self.get_producers_for_subtile_region (
                tpb          = tpb,
                fmap_subtile = ifmap_tile_subtile,
                reader_engine = reader_engine,
                )
    last_waveop = self.gen_dequantize_waveop_inline(
                tpb              = tpb,
                op               = first_op,
                ifmap_tile       = ifmap_tile,
                ofmap_tile       = ofmap_tile,
                src_is_psum      = first_op.src_is_psum,
                psum_bank_src    = psum_bank_id if first_op.src_is_psum else -1,
                dst_is_psum      = first_op.dst_is_psum,
                psum_bank_dst    = psum_bank_id if first_op.dst_is_psum else -1,
                dram_waveops     = dram_ifmaps_waveops)
    attach_predecessors(last_waveop, prev_waveops)
    self.mark_producers_for_subtile_region(
            tpb           = tpb,
            fmap_subtile  = ofmap_tile_subtile,
            waveop        = last_waveop)

# Execute a quantize operator
def execute_quantize_op(self, tpb, batch_item):
    first_op = self[0]
    quant_scale = first_op.data['quant_scale']
    zero_point = first_op.data['zero_point']
    first_op.ofmaps_file_params.dram_data = tpb.pool.quantize_uint8(first_op.ifmaps_file_params.dram_data, quant_scale, zero_point)
    n_id = batch_item // first_op.Tn
    for m_id in range(first_op.m):
        for h_id in range(first_op.h):
            for w_id in range(first_op.w):
                tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False)
                ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = self.first_op.dst_is_psum)
                execute_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                emit_waveops_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)

def execute_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
    first_op = self.first_op
    ifmap_subtile = ifmap_tile.make_pewave()
    ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
    input_tilex = ifmap_tile.tile_rect.dim2d.x
    input_tiley = ifmap_tile.tile_rect.dim2d.y
    ifmaps_data_extract = ifmaps_data[0:input_tiley*input_tilex*first_op.Tn, :]
    assert first_op.data['layer_type'] == 'QuantizeV2'
    quant_scale = self.first_op.data['quant_scale']
    zero_point = self.first_op.data['zero_point']
    tile_data_flatten = tpb.pool.quantize_uint8(ifmaps_data_extract, quant_scale, zero_point)
    if psum_bank_id >= 0:
        tpb.pearray.write_psum(psum_bank_id, 0, tile_data_flatten.shape[0], tile_data_flatten)

def emit_waveops_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
    # wave loop ordering scheme: nmhw
    first_op = self.first_op
    ifmap_tile_subtile \
            = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, Dim2D(1,1))
    ofmap_tile_subtile \
            = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, Dim2D(1,1))
    layer_type = first_op.data['layer_type']
    reader_engine = EngineEnum.POOL
    (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) \
        = self.get_producers_for_subtile_region (
                tpb          = tpb,
                fmap_subtile = ifmap_tile_subtile,
                reader_engine = reader_engine,
                )
    last_waveop = gen_quantize_waveop_inline(self,
                tpb              = tpb,
                op               = first_op,
                ifmap_tile       = ifmap_tile,
                ofmap_tile       = ofmap_tile,
                src_is_psum      = first_op.src_is_psum,
                psum_bank_src    = psum_bank_id if first_op.src_is_psum else -1,
                dst_is_psum      = first_op.dst_is_psum,
                psum_bank_dst    = psum_bank_id if first_op.dst_is_psum else -1,
                dram_waveops     = dram_ifmaps_waveops)
    attach_predecessors(last_waveop, prev_waveops)
    self.mark_producers_for_subtile_region(
            tpb           = tpb,
            fmap_subtile  = ofmap_tile_subtile,
            waveop        = last_waveop)

# generate dequantize instruction and add it to instruction stream
def gen_dequantize_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops):
    batch_item = ofmap_tile.n_id * ofmap_tile.Tn
    layer_name = op.data["layer_name"]
    out_dtype = op.get_out_data_type()
    ifmap_subtile = ifmap_tile.make_pewave()
    src_x_num = ifmap_subtile.subtile_rect.dim2d.x
    src_y_num = ifmap_subtile.subtile_rect.dim2d.y
    src_z_num = ifmap_subtile.tile.Tn
    waveop = {}
    if src_is_psum:
        in_dtype = "int32"
        src_sb_address = 0
        src_y_step = ifmap_subtile.subtile_rect.dim2d.x
        src_z_step = src_y_step * src_y_num if src_z_num > 1 else 1
        waveop["src_psum_bank_id"]      = psum_bank_src
        waveop["src_psum_bank_offset"]  = ifmap_subtile.subtile_psum_offset
    else:
        in_dtype = op.prev[0].get_out_data_type()
        src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
        src_y_step = ifmap_subtile.tile.file_params.file_dims.W
        src_z_step = ifmap_subtile.tile.file_params.batch_item_partition_usage_elems_padded if src_z_num > 1 else 1
        waveop["src_start_at_mid_part"] = ifmap_tile.start_at_mid_part
        waveop["src_sb_address"]        = src_sb_address
    ofmap_subtile = ofmap_tile.make_pewave()
    dst_x_num = ofmap_subtile.subtile_rect.dim2d.x
    dst_y_num = ofmap_subtile.subtile_rect.dim2d.y
    dst_z_num = ofmap_subtile.tile.Tn
    num_partitions = ofmap_tile.channel_count
    if dst_is_psum:
        dst_sb_address = 0
        dst_y_step = ofmap_subtile.subtile_rect.dim2d.x
        dst_z_step = dst_y_step * dst_y_num if dst_z_num > 1 else 1
        waveop["dst_psum_bank_id"]      = psum_bank_dst
        waveop["dst_psum_bank_offset"]  = ofmap_subtile.subtile_psum_offset
    else:
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
        dst_y_step = ofmap_subtile.tile.file_params.file_dims.W
        dst_z_step = ofmap_subtile.tile.file_params.batch_item_partition_usage_elems_padded if dst_z_num > 1 else 1
        waveop["dst_start_at_mid_part"] = ofmap_tile.start_at_mid_part
        waveop["dst_sb_address"]        = dst_sb_address
    waveop_name = layer_name + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
    dequant_scale = op.data['dequant_scale']
    zero_point = op.data['zero_point']
    scalar_val = [-zero_point, dequant_scale]
    scalar_op = [
        self.translate_isa_op_name("add"),
        self.translate_isa_op_name("multiply"),
    ]
    instr = {
        'previous_waveops'        : [],
        'waveop_type'             : 'TensorScalar',
        'waveop_name'             : waveop_name,
        'layer_name'              : layer_name,
        'tile_id_format'          : ofmap_tile.format,
        'tile_id'                 : ofmap_tile.id_array,
        'is_scalar_op'            : True,
        'in_dtype'                : in_dtype,
        'out_dtype'               : out_dtype,
        'src_is_psum'             : src_is_psum,
        'src_x_step'              : 1,
        'src_x_num'               : src_x_num,
        'src_y_step'              : src_y_step,
        'src_y_num'               : src_y_num,
        'src_z_step'              : src_z_step,
        'src_z_num'               : src_z_num,
        'dst_is_psum'             : dst_is_psum,
        'dst_x_step'              : 1,
        'dst_x_num'               : dst_x_num,
        'dst_y_step'              : dst_y_step,
        'dst_y_num'               : dst_y_num,
        'dst_z_step'              : dst_z_step,
        'dst_z_num'               : dst_z_num,
        'num_partitions'          : num_partitions,
        'op0'                     : scalar_op[0],
        'op1'                     : scalar_op[1],
        'imm_val0'                : scalar_val[0],
        'imm_val1'                : scalar_val[1],
        'reverse0'                : False,
        'reverse1'                : False,
        **waveop,
    }
    tpb.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)
    return instr

# generate quantize (float32 --> uint8) instruction and add it to instruction stream
def gen_quantize_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops):
    batch_item = ofmap_tile.n_id * ofmap_tile.Tn
    layer_name = op.data["layer_name"]
    in_dtype = 'float32'
    out_dtype = op.get_out_data_type()
    ifmap_subtile = ifmap_tile.make_pewave()
    src_x_num = ifmap_subtile.subtile_rect.dim2d.x
    src_y_num = ifmap_subtile.subtile_rect.dim2d.y
    src_z_num = ifmap_subtile.tile.Tn
    waveop = {}
    if src_is_psum:
        src_sb_address = 0
        src_y_step = ifmap_subtile.subtile_rect.dim2d.x
        src_z_step = src_y_step * src_y_num if src_z_num > 1 else 1
        waveop["src_psum_bank_id"]      = psum_bank_src
        waveop["src_psum_bank_offset"]  = ifmap_subtile.subtile_psum_offset
    else:
        src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
        src_y_step = ifmap_subtile.tile.file_params.file_dims.W
        src_z_step = ifmap_subtile.tile.file_params.batch_item_partition_usage_elems_padded if src_z_num > 1 else 1
        waveop["src_start_at_mid_part"] = ifmap_tile.start_at_mid_part
        waveop["src_sb_address"]        = src_sb_address
    ofmap_subtile = ofmap_tile.make_pewave()
    dst_x_num = ofmap_subtile.subtile_rect.dim2d.x
    dst_y_num = ofmap_subtile.subtile_rect.dim2d.y
    dst_z_num = ofmap_subtile.tile.Tn
    num_partitions = ofmap_tile.channel_count
    if dst_is_psum:
        dst_sb_address = 0
        dst_y_step = ofmap_subtile.subtile_rect.dim2d.x
        dst_z_step = dst_y_step * dst_y_num if dst_z_num > 1 else 1
        waveop["dst_psum_bank_id"]      = psum_bank_dst
        waveop["dst_psum_bank_offset"]  = ofmap_subtile.subtile_psum_offset
    else:
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
        dst_y_step = ofmap_subtile.tile.file_params.file_dims.W
        dst_z_step = ofmap_subtile.tile.file_params.batch_item_partition_usage_elems_padded if dst_z_num > 1 else 1
        waveop["dst_start_at_mid_part"] = ofmap_tile.start_at_mid_part
        waveop["dst_sb_address"]        = dst_sb_address
    waveop_name = layer_name + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
    quant_scale = op.data['quant_scale']
    zero_point = op.data['zero_point']
    scalar_val = [quant_scale, zero_point]
    scalar_op = [
        self.translate_isa_op_name("multiply"),
        self.translate_isa_op_name("add"),
    ]
    instr = {
        'previous_waveops'        : [],
        'waveop_type'             : 'TensorScalar',
        'waveop_name'             : waveop_name,
        'layer_name'              : layer_name,
        'tile_id_format'          : ofmap_tile.format,
        'tile_id'                 : ofmap_tile.id_array,
        'in_dtype'                : in_dtype,
        'out_dtype'               : out_dtype,
        'src_is_psum'             : src_is_psum,
        'src_x_step'              : 1,
        'src_x_num'               : src_x_num,
        'src_y_step'              : src_y_step,
        'src_y_num'               : src_y_num,
        'src_z_step'              : src_z_step,
        'src_z_num'               : src_z_num,
        'dst_is_psum'             : dst_is_psum,
        'dst_x_step'              : 1,
        'dst_x_num'               : dst_x_num,
        'dst_y_step'              : dst_y_step,
        'dst_y_num'               : dst_y_num,
        'dst_z_step'              : dst_z_step,
        'dst_z_num'               : dst_z_num,
        'num_partitions'          : num_partitions,
        'op0'                     : scalar_op[0],
        'op1'                     : scalar_op[1],
        'imm_val0'                : scalar_val[0],
        'imm_val1'                : scalar_val[1],
        'reverse0'                : False,
        'reverse1'                : False,
        **waveop,
    }
    tpb.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)
    return instr

def mem_pattern_uint8_perf_opt(self, instr, ifmap_tile_dim2d):
    if self.conv_op.data['layer_type'] == 'QuantizedConv':
        mode_possible = set()
        src_xy_num = instr['src_x_num'] * instr['src_y_num']
        src_contiguous = (self.conv_op.stride.x == 1 and
            instr['src_x_num'] == ifmap_tile_dim2d.x and
            instr['src_sb_address'] % 2 == 0 and
            instr['dst_psum_bank_offset'] % 2 == 0 and
            (src_xy_num % 2 == 0 or instr['src_y_num'] == ifmap_tile_dim2d.y))
        if src_contiguous:
            mode_possible.add('double_pixel_contiguous')
        src_even = (self.conv_op.stride.x == 1 and
            instr['src_x_num'] % 2 == 0 and
            instr['src_sb_address'] % 2 == 0 and
            instr['dst_psum_bank_offset'] % 2 == 0 and
            instr['src_y_step'] % 2 == 0 and
            instr['dst_y_step'] % 2 == 0)
        if src_even:
            mode_possible.add('double_pixel_even')
        if 'double_pixel_contiguous' in mode_possible:
            instr['src_x_step'] = instr['dst_x_step'] = 2
            instr['src_x_num']  = instr['dst_x_num']  = (src_xy_num + 1) // 2
            instr['src_y_step'] = instr['dst_y_step'] = src_xy_num
            instr['src_y_num']  = instr['dst_y_num']  = 1
            instr['pe_perf_opt_mode'] = 'double_pixel'
        elif 'double_pixel_even' in mode_possible:
            instr['src_x_step'] = instr['dst_x_step'] = 2
            instr['src_x_num'] //= 2
            instr['dst_x_num'] //= 2
            instr['pe_perf_opt_mode'] = 'double_pixel'
    return instr

def reorder_waveops_uint8_perf_opt(self, waveop_list):
    # need to reorder uint8 matmul's for convolution if performance mode is on
    padding_diff = self.conv_op.padES - self.conv_op.padWN
    padding_condition = padding_diff.x in {0, 1} and padding_diff.y in {0, 1}
    if not (self.conv_op.ifmaps_file_params.data_type == 'uint8' and
            self.conv_op.data['layer_type'] == 'QuantizedConv' and
            self.conv_op.stride == Dim2D(1, 1) and padding_condition):
        return waveop_list
    pos_first_matmul = 0
    for waveop in waveop_list:
        if waveop['waveop_type'] == 'MatMul':
            break
        else:
            pos_first_matmul += 1
    waveops_matmul = []
    for waveop in waveop_list[pos_first_matmul:]:
        if waveop['waveop_type'] == 'MatMul':
            waveops_matmul.append(waveop)
    if len(waveops_matmul) <= 1: # no need to reorder
        return waveop_list
    first_parent_names = waveops_matmul[0]['previous_waveops']
    first_parent_names_matmul = []
    first_parent_names_other = []
    for name in first_parent_names:
        if 'MatMul' in name:
            first_parent_names_matmul.append(name)
        else:
            first_parent_names_other.append(name)

    # find the middle MatMul waveop; it needs to be the first MatMul
    # this finding method most likely only works with 'VALID' or 'SAME' padding
    # that's why we are checking `padding_condition` earlier
    mid_matmul_y = (self.conv_op.weights_file_params.file_dims.R - 1) // 2
    mid_matmul_x = (self.conv_op.weights_file_params.file_dims.S - 1) // 2
    mid_matmul_waveop = waveops_matmul[mid_matmul_y * self.conv_op.weights_file_params.file_dims.R + mid_matmul_x]
    pos_mid_matmul_waveop = pos_first_matmul
    for waveop in waveop_list[pos_first_matmul:]:
        if waveop is mid_matmul_waveop:
            break
        else:
            pos_mid_matmul_waveop += 1
    if pos_mid_matmul_waveop <= pos_first_matmul: # no need to reorder
        return waveop_list
    # classify waveops before the middle MatMul into separate lists
    waveops_matmul_name_set = set()
    waveops_sbatomload = []
    waveops_keep_order = []
    for waveop in waveop_list[pos_first_matmul:pos_mid_matmul_waveop]:
        if waveop['waveop_type'] == 'MatMul':
            waveops_matmul_name_set.add(waveop['waveop_name'])
        if waveop['waveop_type'] == 'SBAtomLoad':
            waveops_sbatomload.append(waveop)
        else:
            waveops_keep_order.append(waveop)

    reordered_waveops = []
    # waveops of type SBAtomLoad that are scheduled before the middle MatMul
    # need to be placed at the very front.
    # Need to make them not depend on MatMul's that are originally before the middle MatMul.
    # Also need to make them depend on waveops that the original first MatMul depends on,
    # here we make a small relaxation to let them depend on only waveops that are not of type MatMul
    for waveop in waveops_sbatomload:
        remove_parent_names_in_set(waveop, waveops_matmul_name_set)
        add_parent_names_in_list(waveop, first_parent_names_other)
        reordered_waveops.append(waveop)

    # The original middle MatMul is placed right after SBAtomLoad waveops.
    # It needs to depend on waveops that the original first MatMul depends on,
    # plus all SBAtomLoad waveops we brought to the front
    new_first_parent_names = first_parent_names.copy()
    for waveop in waveops_sbatomload:
        new_first_parent_names.append(waveop['waveop_name'])
    mid_matmul_waveop['previous_waveops'] = new_first_parent_names
    reorder_parent_names(mid_matmul_waveop)
    reordered_waveops.append(mid_matmul_waveop)

    # The original first MatMul need to depend on the original middle MatMul
    remove_parent_names_of_matmul(waveops_keep_order[0])
    add_parent_names_in_list(waveops_keep_order[0], [mid_matmul_waveop['waveop_name']])
    reorder_parent_names(waveops_keep_order[0])
    reordered_waveops.extend(waveops_keep_order)

    # swap start_tensor_calc for the original first and middle MatMul waveops
    mid_matmul_waveop['start_tensor_calc'], waveops_keep_order[0]['start_tensor_calc'] = \
        waveops_keep_order[0]['start_tensor_calc'], mid_matmul_waveop['start_tensor_calc']

    # replace reordered waveops
    waveop_list[pos_first_matmul:pos_mid_matmul_waveop+1] = reordered_waveops

    # The MatMul after the original middle MatMul need to depend on the new middle MatMul
    # and need not depend on any other MatMul between the original first and the original middle
    remove_parent_names_in_set(waveop_list[pos_mid_matmul_waveop+1],
        {*waveops_matmul_name_set, mid_matmul_waveop['waveop_name']})
    add_parent_names_in_list(waveop_list[pos_mid_matmul_waveop+1], [waveops_keep_order[-1]['waveop_name']])
    reorder_parent_names(waveop_list[pos_mid_matmul_waveop+1])
    return waveop_list

"""Remove parent names from waveop['previous_waveops'] for any name in name_set.
Modifies waveop in place.
"""
def remove_parent_names_in_set(waveop, name_set):
    new_prev_waveops = []
    for name in waveop['previous_waveops']:
        if name not in name_set:
            new_prev_waveops.append(name)
    waveop['previous_waveops'] = new_prev_waveops
    return waveop

"""Aggressively remove any name with MatMul from waveop['previous_waveops'].
Modifies waveop in place.
"""
def remove_parent_names_of_matmul(waveop):
    new_prev_waveops = []
    for name in waveop['previous_waveops']:
        if 'MatMul' not in name:
            new_prev_waveops.append(name)
    waveop['previous_waveops'] = new_prev_waveops
    return waveop

"""Append parent names in name_list into waveop['previous_waveops']
Modifies waveop in place. Assumes that name_list contains no repeating element.
"""
def add_parent_names_in_list(waveop, name_list):
    previous_waveops_set = set(waveop['previous_waveops'])
    for name in name_list:
        if name not in previous_waveops_set:
            waveop['previous_waveops'].append(name)
    return waveop

"""Reorder waveop['previous_waveops'] so that MatMul waveops come first, just for safety.
Modifies waveop in place.
"""
def reorder_parent_names(waveop):
    prev_waveops_matmul = []
    prev_waveops_other = []
    for name in waveop['previous_waveops']:
        if 'MatMul' in name:
            prev_waveops_matmul.append(name)
        else:
            prev_waveops_other.append(name)
    new_prev_waveops = []
    for name in prev_waveops_matmul:
        new_prev_waveops.append(name)
    for name in prev_waveops_other:
        new_prev_waveops.append(name)
    waveop['previous_waveops'] = new_prev_waveops
    return waveop
