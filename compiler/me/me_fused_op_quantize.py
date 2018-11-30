"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Fused operations execution for quantized operators, including verification and waveop generation
"""
from me_fused_op import *


# Execute a dequantize operator
def execute_dequantize_op(self, tpb, batch_item):
    first_op = self[0]
    n_id = batch_item // first_op.Tn
    for m_id in range(first_op.m):
        for h_id in range(first_op.h):
            for w_id in range(first_op.w):
                tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False,
                                    stridedslice_chan_offset = first_op.stridedslice_chan_offset)
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
    ifmap_subtile = ifmap_tile.make_pewave(self.first_op.stridedslice_chan_offset)
    ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
    input_tilex = ifmap_tile.tile_rect.dim2d.x
    input_tiley = ifmap_tile.tile_rect.dim2d.y
    ifmaps_data_extract = ifmaps_data [0:input_tiley*input_tilex*first_op.Tn, :]
    layer_type = first_op.data['layer_type']
    assert layer_type == 'Dequantize'
    dequant_scale = self.first_op.data['dequant_scale']
    zero_point = self.first_op.data['zero_point']
    tile_data_flatten = tpb.pool.dequantize(ifmaps_data_extract, dequant_scale, zero_point)
    assert first_op.dst_is_psum
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

    dequant_scale = first_op.data['dequant_scale']
    zero_point = first_op.data['zero_point']
    last_waveop = self.gen_dequantize_waveop_inline(
                tpb              = tpb,
                op               = first_op,
                ifmap_tile       = ifmap_tile,
                ofmap_tile       = ofmap_tile,
                src_is_psum      = first_op.src_is_psum,
                psum_bank_src    = -1,
                dst_is_psum      = True,
                psum_bank_dst    = psum_bank_id if first_op.dst_is_psum else -1,
                dram_waveops     = dram_ifmaps_waveops,
                dequant_scale    = dequant_scale,
                zero_point       = zero_point)
    attach_predecessors(last_waveop, prev_waveops)
    self.mark_producers_for_subtile_region(
            tpb           = tpb,
            fmap_subtile  = ofmap_tile_subtile,
            waveop        = last_waveop)

# Execute a quantize operator
def execute_quantize_op(self, tpb, batch_item):
    first_op = self[0]
    n_id = batch_item // first_op.Tn
    for m_id in range(first_op.m):
        for h_id in range(first_op.h):
            for w_id in range(first_op.w):
                tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False,
                                    stridedslice_chan_offset = first_op.stridedslice_chan_offset)
                ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = self.first_op.dst_is_psum)
                execute_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                emit_waveops_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id)

def execute_quantize_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
    first_op = self.first_op
    ifmap_subtile = ifmap_tile.make_pewave(self.first_op.stridedslice_chan_offset)
    ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
    input_tilex = ifmap_tile.tile_rect.dim2d.x
    input_tiley = ifmap_tile.tile_rect.dim2d.y
    ifmaps_data_extract = ifmaps_data [0:input_tiley*input_tilex*first_op.Tn, :]
    layer_type = first_op.data['layer_type']
    assert layer_type == 'QuantizeV2'
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

    quant_scale = first_op.data['quant_scale']
    zero_point = first_op.data['zero_point']
    last_waveop = gen_quantize_waveop_inline(self,
                tpb              = tpb,
                op               = first_op,
                ifmap_tile       = ifmap_tile,
                ofmap_tile       = ofmap_tile,
                src_is_psum      = first_op.src_is_psum,
                psum_bank_src    = -1,
                dst_is_psum      = first_op.dst_is_psum,
                psum_bank_dst    = psum_bank_id if first_op.dst_is_psum else -1,
                dram_waveops     = dram_ifmaps_waveops,
                quant_scale      = quant_scale,
                zero_point       = zero_point)
    attach_predecessors(last_waveop, prev_waveops)
    self.mark_producers_for_subtile_region(
            tpb           = tpb,
            fmap_subtile  = ofmap_tile_subtile,
            waveop        = last_waveop)

# generate dequantize instruction and add it to instruction stream
def gen_dequantize_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops, dequant_scale, zero_point):
    batch_item = ofmap_tile.n_id * op.Tn
    layer_name = op.data["layer_name"]
    out_dtype = op.get_out_data_type()
    if (src_is_psum):
        in_dtype = "int32"
        src_sb_address = 0
    else:
        in_dtype = op.prev[0].get_out_data_type()
        src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
    dst_x_num = op.ofmap_full_tilex_sz
    dst_y_step = op.E
    dst_y_num = op.ofmap_full_tiley_sz
    dst_z_step = op.ofmaps_file_params.batch_item_partition_usage_elems_padded if op.Tn > 1 else 1
    dst_z_num = op.Tn  # Need CNHW data format
    num_partitions = ofmap_tile.channel_count
    if dst_is_psum:
        dst_sb_address = 0
    else:
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
    waveop_name = layer_name + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
    instr = {
          'previous_waveops'        : [],
          'waveop_type'             : op.data['layer_type'],
          'waveop_name'             : waveop_name,
          'layer_name'              : layer_name,
          'tile_id_format'          : ofmap_tile.format,
          'tile_id'                 : ofmap_tile.id_array,
          'is_scalar_op'            : True,
          'in_dtype'                : in_dtype,
          'out_dtype'               : out_dtype,
          'src_is_psum'             : src_is_psum,
          'src_x_step'              : 1,
          'src_x_num'               : dst_x_num,
          'src_y_step'              : dst_y_step,
          'src_y_num'               : dst_y_num,
          'src_z_step'              : dst_z_step,
          'src_z_num'               : dst_z_num,
          'dst_is_psum'             : dst_is_psum,
          'dst_x_step'              : 1,
          'dst_x_num'               : dst_x_num,
          'dst_y_step'              : dst_y_step,
          'dst_y_num'               : dst_y_num,
          'dst_z_step'              : dst_z_step,
          'dst_z_num'               : dst_z_num,
          'num_partitions'          : num_partitions,
        }
    instr['waveop_type'] = "ScaleAdd"
    instr['scale'] = dequant_scale
    instr['add'] = -zero_point * dequant_scale
    if src_is_psum:
        assert(psum_bank_src >= 0)
        instr['src_psum_bank_id'] = psum_bank_src
        instr['src_psum_bank_offset'] = 0
    else:
        instr['src_sb_address'] = src_sb_address
        instr['src_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
    if dst_is_psum:
        assert(psum_bank_dst >= 0)
        instr['dst_psum_bank_id'] = psum_bank_dst
        instr['dst_psum_bank_offset'] = 0
    else:
        instr['dst_sb_address'] = dst_sb_address
        instr['dst_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
    tpb.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)
    return instr

# generate quantize (float32 --> uint8) instruction and add it to instruction stream
def gen_quantize_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops, quant_scale, zero_point):
    batch_item = ofmap_tile.n_id * op.Tn
    layer_name = op.data["layer_name"]
    in_dtype = 'float32'
    out_dtype = op.get_out_data_type()
    if src_is_psum:
        src_sb_address = 0
    else:
        src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
    dst_x_num = op.ofmap_full_tilex_sz
    dst_y_step = op.E
    dst_y_num = op.ofmap_full_tiley_sz
    dst_z_step = op.ofmaps_file_params.batch_item_partition_usage_elems_padded if op.Tn > 1 else 1
    dst_z_num = op.Tn  # Need CNHW data format
    num_partitions = ofmap_tile.channel_count
    if dst_is_psum:
        dst_sb_address = 0
    else:
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
    waveop_name = layer_name + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
    instr = {
          'previous_waveops'        : [],
          'waveop_type'             : op.data['layer_type'],
          'waveop_name'             : waveop_name,
          'layer_name'              : layer_name,
          'tile_id_format'          : ofmap_tile.format,
          'tile_id'                 : ofmap_tile.id_array,
          'is_scalar_op'            : True,
          'in_dtype'                : in_dtype,
          'out_dtype'               : out_dtype,
          'src_is_psum'             : src_is_psum,
          'src_x_step'              : 1,
          'src_x_num'               : dst_x_num,
          'src_y_step'              : dst_y_step,
          'src_y_num'               : dst_y_num,
          'src_z_step'              : dst_z_step,
          'src_z_num'               : dst_z_num,
          'dst_is_psum'             : dst_is_psum,
          'dst_x_step'              : 1,
          'dst_x_num'               : dst_x_num,
          'dst_y_step'              : dst_y_step,
          'dst_y_num'               : dst_y_num,
          'dst_z_step'              : dst_z_step,
          'dst_z_num'               : dst_z_num,
          'num_partitions'          : num_partitions,
        }
    instr['waveop_type'] = "ScaleAdd"
    instr['scale'] = quant_scale
    instr['add'] = zero_point
    if src_is_psum:
        assert(psum_bank_src >= 0)
        instr['src_psum_bank_id'] = psum_bank_src
        instr['src_psum_bank_offset'] = 0
    else:
        instr['src_sb_address'] = src_sb_address
        instr['src_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
    if dst_is_psum:
        assert(psum_bank_dst >= 0)
        instr['dst_psum_bank_id'] = psum_bank_dst
        instr['dst_psum_bank_offset'] = 0
    else:
        instr['dst_sb_address'] = dst_sb_address
        instr['dst_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
    tpb.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)
    return instr
