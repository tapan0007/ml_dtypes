#include "sequencer.h"
#include "types.h"
#include "string.h"
#include "isa.h"

/*------------------------------------
 * EdgeSignalsInstruction
 *------------------------------------ */
extern addr_t psum_buffer_base;
template<>
void DynamicInstruction<EdgeSignals>::dump(bool header)
{   
    if (header) {
        printf("rc cc |  iv   ia  |  wv wa   wd  wt wc | pa ps pe | av af | pv pt px py  \n");
    }
    printf("%2d %2d | %2d 0x%-3lx  | %2d 0x%-3lx  %2d %2d %2d | 0x%-3lx %2d %2d | %2d %2d | %2d %2d %2d %2d  \n",
            args.row_countdown, args.column_countdown, 
            args.ifmap_valid, args.ifmap_full_addr,
            args.weight_valid, args.weight_full_addr, args.weight_dtype, args.weight_toggle, args.weight_clamp,
            args.psum_full_addr, args.psum_start, args.psum_stop,
            args.activation_valid, args.activation,
            args.pool_valid, args.pool_type, args.pool_dimx, args.pool_dimy);

}
 
template<>
void
DynamicInstruction<EdgeSignals>::execute(Sequencer *seq) {

    seq->es = args;
    seq->raw_signal = true;
}


/*------------------------------------
 * LdWeightsInstruction
 *------------------------------------ */
template<>
void  DynamicInstruction<LdWeightsArgs>::execute(Sequencer *seq) {
    uint8_t num_cols = args.x_num * args.y_num; 
    seq->es.weight_valid = true;
    seq->es.weight_dtype = (ARBPRECTYPE) args.dtype;
    //seq->es.weight_full_addr = args.address;
    seq->weight_base = args.address;
    seq->es.weight_clamp = (num_cols == 1);
    seq->es.row_countdown = args.num; 
    seq->weight_clamp_countdown = num_cols;
    seq->weight_x_step = args.x_step;
    seq->weight_y_step = args.y_step;
    seq->weight_x_num = args.x_num;
    seq->weight_y_num = args.y_num;
    seq->weight_x_cnt = 0;
    seq->weight_y_cnt = 0;
    seq->raw_signal = false;
    seq->es.weight_full_addr = args.address + (args.y_num * args.y_step - 1) * 
        sizeofArbPrecType((ARBPRECTYPE)args.dtype);

}

/*------------------------------------
 * Convolve
 *------------------------------------ */
template<>
void  DynamicInstruction<MatMulArgs>::execute(Sequencer *seq) {
    /* ifmap setup */
    seq->es.ifmap_full_addr    = args.fmap_start_addr;
    seq->es.ifmap_dtype = (ARBPRECTYPE) args.dtype;
    seq->es.row_countdown = args.num_rows; 
    seq->es.weight_toggle = true;
    seq->ifmap_base = args.fmap_start_addr;
    seq->ifmap_x_num = args.fmap_x_num;
    seq->ifmap_y_num = args.fmap_y_num;
    seq->ifmap_x_cnt = 0;
    seq->ifmap_y_cnt = 0;
    seq->ifmap_x_step = args.fmap_x_step;
    seq->ifmap_y_step = args.fmap_y_step;
    seq->raw_signal = false;

    /* psum setup */
    seq->es.column_countdown = args.num_cols;
    seq->es.psum_start = args.start_tensor_calc;
    seq->es.psum_stop = args.stop_tensor_calc;
    seq->es.psum_full_addr = args.psum_start_addr;

    /* pool setup - TODO remove */
    seq->es.pool_valid = args.stop_tensor_calc; // FIXME - temporary hack

    /* signals that will stay constant for entire convolution */
    /* padding setup */
    seq->pad[N] = args.n_pad;
    seq->pad[S] = args.s_pad;
    seq->pad[E] = args.e_pad;
    seq->pad[W] = args.w_pad;
    /* pifmap is padded ifmap */
    seq->pifmap_x_cnt = 0;
    seq->pifmap_y_cnt = 0;
    seq->pifmap_x_num = seq->ifmap_x_num + args.e_pad + args.w_pad;
    seq->pifmap_y_num = seq->ifmap_y_num + args.n_pad + args.s_pad;
    seq->es.pad_valid = seq->pad_valid(0, 0);
    seq->es.ifmap_valid   = !seq->es.pad_valid;

}

/*------------------------------------
 * Pool
 *------------------------------------ */
template<>
void  DynamicInstruction<PoolArgs>::execute(Sequencer *seq) {
    POOLFUNC pool_func = (POOLFUNC)args.pool_func;
	seq->pool_valid = true;
	seq->pool_timer = Constants::rows;
    seq->ps.func = pool_func;
	seq->ps.dtype = (ARBPRECTYPE)args.in_dtype;
	seq->ps.src_full_addr = args.src_start_addr;
	seq->ps.start = true;
	seq->ps.stop = (pool_func = IDENTITY_POOL) ||
        (args.src_x_num + args.src_y_num == 2);
	seq->ps.dst_full_addr = args.dst_start_addr;
	seq->ps.countdown = args.dst_num;
	
	seq->pool_src_base = args.src_start_addr;
	seq->pool_dst_base = args.dst_start_addr;
	seq->pool_src_x_cnt = 0;
	seq->pool_src_y_cnt = 0;
	seq->pool_src_z_cnt = 0;
	seq->pool_src_x_step = args.src_x_step;
	seq->pool_src_y_step = args.src_y_step;
	seq->pool_src_z_step = args.src_z_step;
	seq->pool_src_x_num = args.src_x_num;
	seq->pool_src_y_num = args.src_y_num;
	seq->pool_src_z_num = args.src_z_num;

	seq->pool_dst_x_step = args.dst_x_step;
	seq->pool_dst_y_step = args.dst_y_step;
	seq->pool_dst_x_cnt = 0;
	seq->pool_dst_y_cnt = 0;
	seq->pool_dst_x_num = args.dst_x_num;
	seq->pool_dst_y_num = args.dst_y_num;
}

/*------------------------------------
 * Sequencer
 *------------------------------------ */
bool
Sequencer::synch() {
    return es.pad_valid || es.ifmap_valid  || es.weight_valid || pool_valid;
}


/* tells if you element (r,c) is a pad (!ifmap) element */
bool
Sequencer::pad_valid(uint8_t r, uint8_t c) {
    /* checks range, using outer ring of padding */
    return r < pad[N] || 
        ((r > (ifmap_y_num + pad[N] - 1)) && 
         (r < ifmap_y_num + pad[N] + pad[S])) ||
        c < pad[W] || 
        ((c > (ifmap_x_num + pad[W] - 1) && 
          (c < ifmap_x_num + pad[W] + pad[E])));
}


void
Sequencer::increment_and_rollover(uint8_t &cnt, uint8_t num, 
        uint8_t &rollover) {
    cnt++;
    if (cnt >= num) {
        cnt = 0;
        rollover++;
    }
}

#define COND_SET(X, VAL) (X == VAL) ? false : X=VAL

/* sub function of step - to step the edgesignal */
void
Sequencer::step_edgesignal() {
    /* UPDATE SEQUENCER STATE */
    if (es.pad_valid) {
        increment_and_rollover(pifmap_x_cnt, pifmap_x_num, pifmap_y_cnt);
    } 
    if (es.ifmap_valid) {
        increment_and_rollover(ifmap_x_cnt, ifmap_x_num, ifmap_y_cnt);
        increment_and_rollover(pifmap_x_cnt, pifmap_x_num, pifmap_y_cnt);
    }
    if (es.weight_valid) {
        increment_and_rollover(weight_x_cnt, weight_x_num, weight_y_cnt);
    }

    /* UPDATE SIGNALS */
    /* IFMAP/PAD */
    /* is the pad/ifmap valid or are we done? */
    es.pad_valid = pad_valid(pifmap_y_cnt, pifmap_x_cnt);
    if (!es.pad_valid && 
            (pifmap_y_cnt == pifmap_y_num)) { 
        /* clear state */
        es.ifmap_valid = false;
        es.psum_start = false;
        es.psum_stop = false;
        es.pool_valid = false;
        es.activation_valid = false;
    } else {
        es.ifmap_valid = !es.pad_valid;
    }

    /* figured out pad/ifmap valid, now compute addresses */
    if (es.ifmap_valid) {
        es.ifmap_full_addr = ifmap_base + 
            (ifmap_y_cnt * ifmap_y_step + ifmap_x_cnt * ifmap_x_step) * 
            sizeofArbPrecType((ARBPRECTYPE)es.ifmap_dtype);
    }
    /* Sending an ifmap, must be getting out ofmap! */
    if (es.ifmap_valid || es.pad_valid) {
        /* FIXME - this is the psum granularity, FIX */
        es.psum_full_addr += Constants::psum_buffer_width;
    }

    /*  WEIGHT */
    /* always toggle down weight */
    COND_SET(es.weight_toggle, false); 
    if (es.weight_valid) {
        if ((weight_x_cnt == weight_x_num - 1) &&
                (weight_y_cnt == weight_y_num -1)) {
            es.weight_clamp = true;
        } else if (weight_y_cnt ==  weight_y_num) {
            assert(es.weight_clamp);
            es.weight_clamp = false;
            es.weight_valid = false;
        } 
        if (es.weight_valid) {
            es.weight_full_addr = weight_base + 
                (weight_y_num * weight_y_step -
                 weight_y_cnt * weight_y_step -
                 weight_x_cnt * weight_x_step - 1) * 
                sizeofArbPrecType((ARBPRECTYPE)es.weight_dtype);
        }
    }
}

/* sub function of step - to step the poolsignal */
void
Sequencer::step_poolsignal() {
    if (!pool_valid) {
        return;
    }
    if (pool_timer) {
        --pool_timer;
        if (pool_timer == 0) {
            ps.valid = true;
        } 
        return;
    }
    assert(ps.valid);
    if (!ps.valid) {
        return;
    }
    size_t dsize = sizeofArbPrecType((ARBPRECTYPE)ps.dtype);
    /* roll over src counters */
    pool_src_x_cnt++;
    if (pool_src_x_cnt >= pool_src_x_num) {
        pool_src_x_cnt = 0;
        pool_src_y_cnt++;
    }

    if (pool_src_y_cnt >= pool_src_y_num) {
        pool_src_y_cnt = 0;
        pool_src_z_cnt++;
    }

    if (pool_src_z_cnt >= pool_src_z_num) {
        pool_src_z_cnt = 0;
    }
    bool eopool = 
        (pool_src_x_cnt == pool_src_x_num - 1) && 
        (pool_src_y_cnt == pool_src_y_num - 1) &&
        (pool_src_z_cnt == pool_src_z_num - 1);

    /* roll over dst counters */
    if (ps.func == IDENTITY_POOL || eopool) {
        pool_dst_x_cnt++;
        if (pool_dst_x_cnt >= pool_dst_x_num) {
            pool_dst_x_cnt = 0;
            pool_dst_y_cnt++;
        }
    }

    /* set signals */
    /* togglers */
    COND_SET(ps.stop, false);
    COND_SET(ps.start, false);

    /* start/stop pooling */
    if (ps.func == IDENTITY_POOL) {
        ps.start = true;
        ps.stop = true;
    } else {
        ps.start = ps.stop; /* stopped last cycle, start anew */
        ps.stop = eopool;
    }

    /* totally done */
    if (pool_dst_y_cnt >= pool_dst_y_num) {
        pool_dst_y_cnt = 0;
        ps.valid = false;
        pool_valid = false;
    }


    /* calculate address based on settings */
    if (ps.valid) {
        ps.src_full_addr = pool_src_base + 
            (pool_src_z_cnt * pool_src_z_step +
             pool_src_y_cnt * pool_src_y_step + 
             pool_src_x_cnt * pool_src_x_step) * dsize;
    }
    if (ps.start) {
        ps.dst_full_addr = pool_dst_base + 
            (pool_dst_y_cnt * pool_dst_y_step + 
             pool_dst_x_cnt * pool_dst_x_step) * dsize;
    }
}

void
Sequencer::step() {
    /* empty the instruction queue */
    if (raw_signal || !synch()) {
        if (!uop.empty()) {
            Instruction *inst = uop.front();
            inst->execute(this);
            uop.pop();
            free(inst);
        } else {
            es = {0};
        }
        return;
    }
    /* was the instruction a raw signal, if so, leave es alone and exit */
    if (raw_signal) {
        return;
    }

    step_edgesignal();
    step_poolsignal();

    clock++;
}


EdgeSignals Sequencer::pull_edge() {
    return es;
}

PoolSignals
Sequencer::pull_pool() {
    return ps;
}

void
Sequencer::dump() {
    int num = uop.size();
    Instruction *insn;
    for (int i = 0; i < num; i++) {
        insn = uop.front();
        insn->dump(!i);
        uop.pop();
        uop.push(insn);
    }
}


#define PUSH push

unsigned int
Sequencer::get_tile_type(unsigned int row, unsigned int col, 
        unsigned int n_rows, unsigned int n_cols) {
    unsigned int tt = 0;
    if (row == 0) {
        tt |= N_FLAG;
    }
    if (row == n_rows - 1) {
        tt |= S_FLAG;
    }
    if (col == 0) {
        tt |= W_FLAG;
    }
    if (col == n_cols - 1) {
        tt |= E_FLAG;
    }
    return tt;
}

void 
Sequencer::convolve_dynamic(const ConvolveArgs &args, unsigned int &o_rows,
        unsigned int &o_cols)
{
    unsigned int f_rows = args.w_r;
    unsigned int f_cols = args.w_s;
    unsigned int i_rows = args.i_h;
    unsigned int i_cols = args.i_w;
    //unsigned int p_rows = args.padding_rows;
    //unsigned int p_cols = args.padding_cols;
    unsigned int p_rows = args.padding_rows;
    unsigned int p_cols = args.padding_cols;
    unsigned int d_rows = 0; // TODO: implement
    unsigned int d_cols = 0; // TODO: implement
    unsigned int s_rows = 1; // TODO: implement
    unsigned int s_cols = 1; // TODO: implement
    unsigned int f_rows_dilated = f_rows + (f_rows - 1) * d_rows;
    unsigned int f_cols_dilated = f_cols + (f_cols - 1) * d_cols;
    /* for output dim derivation, see https://arxiv.org/pdf/1603.07285.pdf */
    o_rows = (i_rows - f_rows_dilated + 2 * p_rows) / s_rows + 1;
    o_cols = (i_cols - f_cols_dilated + 2 * p_cols) / s_cols + 1;
    unsigned int num_cols = args.w_m;
    unsigned int num_rows = args.i_c;
    unsigned int tile_rows = ceil((float) o_rows / Constants::tile_size);
    unsigned int tile_cols = ceil((float) o_rows / Constants::tile_size);
    ARBPRECTYPE ifmap_dtype = UINT8;
    ARBPRECTYPE weight_dtype = UINT8;
    addr_t weight_dsize = sizeofArbPrecType(weight_dtype);
    addr_t ifmap_full_addr = args.ifmap_full_addr;
    addr_t ofmap_full_addr = args.ofmap_full_addr;
    size_t dsize = sizeofArbPrecType((ARBPRECTYPE)ifmap_dtype);
    addr_t weight_step;
    LdWeightsArgs weight_args;
    MatMulArgs    matmul_args;
    PoolArgs      pool_args;

    /* weight args */
    weight_args.dtype = weight_dtype;
    weight_args.num = num_rows;
    weight_args.x_step = weight_dsize;
    weight_args.x_num = args.w_m;
    weight_args.y_step = weight_dsize * args.w_m;
    weight_args.y_num = 1;
    weight_args.address = args.filter_full_addr;
    weight_step = weight_args.y_num * weight_args.y_step * weight_dsize;

    /* matmul args */
    matmul_args.fmap_x_step = 1;
    matmul_args.fmap_y_step = i_cols;
    matmul_args.dtype = ifmap_dtype;
    matmul_args.psum_start_addr = 0; /* b/c we specify padding as arg */
    matmul_args.num_rows = num_rows;
    matmul_args.num_cols = num_cols;

    /* pool args */
    ARBPRECTYPE pool_dtype = UINT32;
    addr_t pool_dsize = sizeofArbPrecType((ARBPRECTYPE)pool_dtype);
    pool_args.pool_func = IDENTITY_POOL;
    pool_args.in_dtype     = pool_dtype;
    pool_args.src_start_addr = psum_buffer_base;
    pool_args.src_x_step= 1;
    pool_args.src_y_step= 1;
    pool_args.src_z_step= 1;
    pool_args.src_y_num = 1;
    pool_args.src_z_num = 1;
    pool_args.dst_x_step = 1;
    pool_args.dst_y_step = o_cols;
    pool_args.dst_start_addr = ofmap_full_addr;
    pool_args.dst_num = args.w_m;

    /* tile args */
    size_t tile_x_dim = Constants::tile_size;
    size_t tile_y_dim = Constants::tile_size;
    size_t tile_x_whole = o_cols > tile_x_dim ? tile_x_dim : o_cols;
    size_t tile_y_whole = o_rows > tile_y_dim ? tile_y_dim : o_rows;
    size_t tile_x_partial = 
        o_cols % tile_x_dim ? o_cols % tile_x_dim : tile_x_whole;
    size_t tile_y_partial = 
        o_rows % tile_y_dim ? o_rows % tile_y_dim : tile_y_whole;

    unsigned int row_offset, col_offset;
    size_t tile_sz_x, tile_sz_y;
    pool_args.dst_start_addr = ofmap_full_addr;
    unsigned int curr_weight = 0;
    unsigned int tt;
    unsigned int r_adj, s_adj;

    /* go through each tile */
    for (unsigned int i = 0; i < tile_rows; i++) {
        for (unsigned int j = 0; j < tile_cols; j++) {
            curr_weight = 0;

            /* load weights ahead of first convolution for first filter! */
            uop.PUSH(new DynamicInstruction<LdWeightsArgs>(weight_args));

            /* go through each weight in the filter and apply it to the ofmap
             * pixels it operates on */
            tt = get_tile_type(i, j, tile_rows, tile_cols);
            tile_sz_x = tt & E_FLAG ? tile_x_partial : tile_x_whole;
            tile_sz_y = tt & S_FLAG ? tile_y_partial : tile_y_whole;

            matmul_args.w_pad = tt & W_FLAG ? p_cols : 0;
            matmul_args.e_pad = tt & E_FLAG ? p_cols : 0;
            matmul_args.n_pad = tt & N_FLAG ? p_rows : 0;
            matmul_args.s_pad = tt & S_FLAG ? p_rows : 0;
            matmul_args.fmap_x_num = tile_sz_x - matmul_args.w_pad -
                matmul_args.e_pad;
            matmul_args.fmap_y_num = tile_sz_y - matmul_args.n_pad - 
                matmul_args.s_pad;
            for (unsigned int r = 0; r <  f_rows; r++) {
                for (unsigned int s = 0; s < f_cols; s++, curr_weight++) {
                    /* matmul arguments and PUSH */
                    r_adj = (r >= p_rows) * (r - p_rows);
                    s_adj = (s >= p_cols) * (s - p_cols);
                    row_offset = (r_adj + i * tile_y_dim); 
                    col_offset = (s_adj + j * tile_x_dim);
                    matmul_args.fmap_start_addr = ifmap_full_addr +
                        (row_offset * i_cols + col_offset) * dsize;

                    matmul_args.start_tensor_calc = (r == 0 && s == 0);
                    matmul_args.stop_tensor_calc = (curr_weight == 
                            (f_rows * f_cols - 1));
                    uop.PUSH(new DynamicInstruction<MatMulArgs>(matmul_args));

                    /* adjust weight address and LdWeights if not at end*/
                    if ((r == f_rows - 1) && (s == f_cols - 1)) {
                        weight_args.address = args.filter_full_addr;
                    } else {
                        weight_args.address += weight_step;
                        uop.PUSH(new DynamicInstruction<LdWeightsArgs>(
                                    weight_args));
                    }
                }
            }
            /* Pool  */
            pool_args.src_x_num = tile_sz_x * tile_sz_y;
            pool_args.dst_x_num = tile_sz_x;
            pool_args.dst_y_num = tile_sz_y;
            uop.PUSH(new DynamicInstruction<PoolArgs>(pool_args));
            if (j < (tile_cols - 1)) { /* non-edge tile */
                pool_args.dst_start_addr += tile_sz_x * pool_dsize;
            } else { /* edge tile */
                pool_args.dst_start_addr = args.ofmap_full_addr +
                    (j * tile_x_whole * tile_y_whole + 
                     tile_sz_x * tile_sz_y) *  pool_dsize;
            }
        }
    }

}

/* Builds recipe for convolution, pushing signals to the end of a "uop" queue
 * which will be read each cycle */
/* TODO: 
 * - ofmap that doesn't fit in psum buffer
 * - pooling
 * - activation
 * - padding
 * - dilation
 * - striding
 * - non-uint8   */
void 
Sequencer::convolve_static(const ConvolveArgs &args)
{
    EdgeSignals es = {};

    int filter_rows = args.w_r;
    int filter_cols = args.w_s;
    int ifmap_rows = args.i_h;
    int ifmap_cols = args.i_w;
    int ofmap_rows = ifmap_rows - filter_rows + 1;
    int ofmap_cols =  ifmap_cols - filter_cols + 1;
    int num_ofmaps = args.w_m;
    int ifmap_channels = args.i_c;
    ARBPRECTYPE ifmap_dtype = UINT8;
    int weight_load_latency = num_ofmaps;
    int curr_opixel, curr_weight;
    int weight_step   = sizeofArbPrecType(args.weight_dtype);
    assert(weight_load_latency < 64 && "Tiling not implemented yet, don't have enough columsn for # ofmaps!");

    printf("warning: bit rot ahead\n");
    /* signals that will stay constant for entire convolution */
    es.weight_dtype  = args.weight_dtype;
    es.activation    = IDENTITY; 
    es.pool_type     = IDENTITY_POOL;
    es.row_countdown = ifmap_channels; 
    es.column_countdown = num_ofmaps;


    /* LOAD WEIGHTS: 
     * first loading of weights for all ofmaps, weight_clamp on last weight */
    es.weight_valid  = true;
    es.weight_full_addr = args.filter_full_addr + (weight_load_latency -1) * sizeofArbPrecType(es.weight_dtype);
    for (int i = 0; i < weight_load_latency - 1; i++) {
        uop.PUSH(new DynamicInstruction<EdgeSignals>(es));
        es.weight_full_addr -= sizeofArbPrecType(es.weight_dtype);
    }
    es.weight_clamp = true;
    uop.PUSH(new DynamicInstruction<EdgeSignals>(es));

    /* LOAD PIXELS:
     * load pixels, load weights in background, compute */
    es.weight_valid = false;
    es.ifmap_valid  = true;
    /* go through each weight in the filter */
    for (int r = 0; r <  filter_rows; r++) {
        for (int s = 0; s < filter_cols; s++) {
            curr_weight = r * filter_cols + s;
            /* go through each ofmap pixel this weight operates one*/
            for (int e = 0; e < ofmap_rows; e++) {
                for (int f = 0; f < ofmap_cols; f++) {
                    curr_opixel = e * ofmap_cols + f; // curr pixel
                    es.psum_full_addr  = curr_opixel * Constants::psum_buffer_width; // FIXME - inefficient use of psumb uffer
                    es.psum_start = (r == 0) && (s == 0); // only clear psum buffer on first weight
                    es.weight_toggle = (e == 0) && (f == 0); // only toggle on first ofmap pixel

                    /* LOAD PIXEL */
                    es.ifmap_full_addr = args.ifmap_full_addr +
                        ((r * ifmap_cols + s) + (e * ifmap_cols) + f) * 
                        sizeofArbPrecType((ARBPRECTYPE)ifmap_dtype);


                    /* LOAD WEIGHTS */
                    if (curr_opixel == 0) {
                        /* we are calculating the weight addr of the next set
                         * (+1) and then trying to get to the last weight (+1)
                         * so we can load the weights in reverse */
                        es.weight_full_addr = args.filter_full_addr + (curr_weight + 1 + 1) * weight_load_latency 
                            * sizeofArbPrecType(es.weight_dtype) - 1;
                        es.weight_valid = true;
                    } else if (curr_opixel < weight_load_latency) {
                        assert(es.weight_valid);
                        es.weight_full_addr -= weight_step;
                    } else if (curr_opixel == weight_load_latency) {
                        es.weight_valid  = false;
                    }
                    /* clamp on last load of weights, or the last ofmap pixel */
                    es.weight_clamp  = (curr_opixel == weight_load_latency - 1);


                    /* ACTIVATE, POOL, WRITE OUT */
                    if ((r == filter_rows - 1) && 
                            (s == filter_cols - 1)) {
                        es.psum_stop   = true;
                        es.activation_valid = true;
                        es.pool_valid = true;
                        //es.ofmap_full_addr = args.ofmap_full_addr + curr_opixel * sizeofArbPrecType(psum_dtype);
                    }
                    
                    uop.PUSH(new DynamicInstruction<EdgeSignals>(es));
                }
            }
        }
    }

    dump();
}

bool
Sequencer::done() {
    return uop.empty() && !es.ifmap_valid && !es.weight_valid && !pool_valid;
}
