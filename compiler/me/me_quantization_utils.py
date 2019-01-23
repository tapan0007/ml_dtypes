"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

from me_utils import Dim2D


def mem_pattern_uint8_perf(conv_op, instr, ifmap_tile_dim2d):
    if conv_op.data['layer_type'] == 'QuantizedConv':
        mode_possible = set()
        src_xy_num = instr['src_x_num'] * instr['src_y_num']
        src_contiguous = (conv_op.stride.x == 1 and
            instr['src_x_num'] == ifmap_tile_dim2d.x and
            instr['src_sb_address'] % 2 == 0 and
            instr['dst_psum_bank_offset'] % 2 == 0 and
            (src_xy_num % 2 == 0 or instr['src_y_num'] == ifmap_tile_dim2d.y))
        if src_contiguous:
            mode_possible.add('double_pixel_contiguous')
        src_even = (conv_op.stride.x == 1 and
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

def reorder_waveops_uint8_perf(conv_op, waveop_list):
    need_reorder = False
    for waveop in waveop_list:
        if 'pe_perf_opt_mode' in waveop and waveop['pe_perf_opt_mode'] != 'none':
            need_reorder = True
            break
    if not need_reorder:
        return waveop_list
    # need to reorder uint8 matmul's for convolution if performance mode is on
    padding_diff = conv_op.padES - conv_op.padWN
    padding_condition = padding_diff.x in {0, 1} and padding_diff.y in {0, 1}
    if not (conv_op.ifmaps_file_params.data_type == 'uint8' and
            conv_op.data['layer_type'] == 'QuantizedConv' and
            conv_op.stride == Dim2D(1, 1) and padding_condition):
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
    mid_matmul_y = (conv_op.weights_file_params.file_dims.R - 1) // 2
    mid_matmul_x = (conv_op.weights_file_params.file_dims.S - 1) // 2
    mid_matmul_waveop = waveops_matmul[mid_matmul_y * conv_op.weights_file_params.file_dims.R + mid_matmul_x]
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
