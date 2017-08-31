#include "sequencer.h"
#include "types.h"
#include "string.h"

/*------------------------------------
 * EdgeSignalsInstruction
 *------------------------------------ */
extern addr_t psum_buffer_base;
template<>
void DynamicInstruction<EdgeSignals>::dump(bool header)
{   
    if (header) {
        printf("rc cc |  iv   ia  |  wv wa   wd  wt wc | pa pd ps pe | av af | pv pt px py pd  \n");
    }
    printf("%2d %2d | %2d 0x%-3lx  | %2d 0x%-3lx  %2d %2d %2d | 0x%-3lx %2d %2d %2d | %2d %2d | %2d %2d %2d %2d %2d   \n",
            args.row_countdown, args.column_countdown, 
            args.ifmap_valid, args.ifmap_full_addr,
            args.weight_valid, args.weight_full_addr, args.weight_dtype, args.weight_toggle, args.weight_clamp,
            args.psum_full_addr, args.psum_dtype, args.psum_start, args.psum_stop,
            args.activation_valid, args.activation,
            args.pool_valid, args.pool_type, args.pool_dimx, args.pool_dimy, args.pool_dtype);

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
    uint8_t dtype_size = sizeofArbPrecType((ARBPRECTYPE)args.dtype);
    seq->es.weight_valid = true;
    seq->es.weight_dtype = (ARBPRECTYPE) args.dtype;
    seq->es.weight_full_addr = args.weight_full_addr;
    seq->es.weight_clamp = (args.num_cols == 1);
    seq->es.row_countdown = args.num_rows; 
    seq->weight_clamp_countdown = args.num_cols;
    seq->weight_x_step = dtype_size * args.x_step;
    seq->weight_x_num = args.x_num;
    seq->weight_y_num = args.y_num;
    seq->weight_x_cnt = 0;
    seq->weight_y_cnt = 0;
    seq->weight_y_step = args.y_step;
    seq->raw_signal = false;

}

/*------------------------------------
 * Convolve
 *------------------------------------ */
template<>
void  DynamicInstruction<MatMulArgs>::execute(Sequencer *seq) {
    /* signals that will stay constant for entire convolution */
    seq->es.ifmap_valid   = true;
    seq->es.ifmap_full_addr    = args.ifmap_full_addr;
    seq->es.ifmap_dtype = (ARBPRECTYPE) args.dtype;
    //seq->es.ofmap_full_addr    = args.ofmap_full_addr;
    seq->es.row_countdown = args.num_rows; 
    seq->es.column_countdown = args.num_cols;
    seq->es.psum_start = args.psum_start;
    seq->es.psum_stop = args.psum_stop;
    seq->es.psum_dtype = (ARBPRECTYPE)args.psum_dtype;
    seq->es.pool_valid = args.psum_stop; // FIXME - temporary hack
    seq->es.pool_dtype = (ARBPRECTYPE)args.psum_dtype; // FIXME - temporary hack to get results
    seq->es.psum_full_addr = 0; // tmp
    seq->es.weight_toggle = true;
    seq->ifmap_base = args.ifmap_full_addr;
    seq->ifmap_x_num = args.x_num;
    seq->ifmap_y_num = args.y_num;
    seq->ifmap_x_cnt = 0;
    seq->ifmap_y_cnt = 0;
    seq->ifmap_x_step = args.x_step;
    seq->ifmap_y_step = args.y_step;
    seq->raw_signal = false;

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
	seq->ps.dtype = (ARBPRECTYPE)args.dtype;
	seq->ps.src_full_addr = args.src_full_addr;
	seq->ps.start = true;
	seq->ps.stop = (pool_func = IDENTITY_POOL) ||
        (args.src_x_num + args.src_y_num == 2);
	seq->ps.dst_full_addr = args.dst_full_addr;
	seq->ps.countdown = args.num_partitions;
	
	seq->pool_src_base = args.src_full_addr;
	seq->pool_dst_base = args.dst_full_addr;
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

Sequencer::Sequencer() : es(), ps(), pool_valid(false), raw_signal(false), clock(0) {
}

Sequencer::~Sequencer() {
}

bool
Sequencer::synch() {
    return es.ifmap_valid  || es.weight_valid || pool_valid;
}

#define COND_SET(X, VAL) (X == VAL) ? false : X=VAL

/* sub function of step - to step the edgesignal */
void
Sequencer::step_edgesignal() {
    /* update state - feed pixel */
    if (es.ifmap_valid) {
        /* es state */
        ifmap_x_cnt++;
        if (ifmap_x_cnt >= ifmap_x_num) {
            ifmap_x_cnt = 0;
            ifmap_y_cnt++;
        }

        if (ifmap_y_cnt == ifmap_y_num) {
            /* clear state */
            es.ifmap_valid = false;
            es.psum_start = false;
            es.psum_stop = false;
            es.pool_valid = false;
            es.activation_valid = false;
        } else {
            es.ifmap_full_addr = ifmap_base + (ifmap_y_cnt * ifmap_y_step + ifmap_x_cnt * ifmap_x_step) * sizeofArbPrecType((ARBPRECTYPE)es.ifmap_dtype);
            es.psum_full_addr += Constants::psum_buffer_width; /* FIXME - this is the psum granularity, we want to use psum buffers more efficiently */
            //es.ofmap_full_addr += sizeofArbPrecType(es.psum_dtype);
        }
        COND_SET(es.weight_toggle, false);
    }
    /* update state - feed weight */
    if (es.weight_valid) {
        /* es state */
        if (--weight_clamp_countdown) {
            es.weight_full_addr -= weight_y_num * weight_y_step;
        }
        if (weight_clamp_countdown == 1) {
            es.weight_clamp = true;
        } else if (weight_clamp_countdown == 0) {
            assert(es.weight_clamp);
            es.weight_clamp = false;
            es.weight_valid = false;
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


EdgeSignals
Sequencer::pull_edge() {
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


void 
Sequencer::convolve_dynamic(const ConvolveArgs &args)
{
    unsigned int filter_rows = args.w_r;
    unsigned int filter_cols = args.w_s;
    unsigned int ifmap_rows = args.i_h;
    unsigned int ifmap_cols = args.i_w;
    unsigned int ofmap_rows = ifmap_rows - filter_rows + 1;
    unsigned int ofmap_cols =  ifmap_cols - filter_cols + 1;
    unsigned int num_cols = args.w_m;
    unsigned int num_rows = args.i_c;
    unsigned int tile_rows = ceil((float) ofmap_rows / Constants::tile_size);
    unsigned int tile_cols = ceil((float) ofmap_rows / Constants::tile_size);
    ARBPRECTYPE ifmap_dtype = UINT8;
    ARBPRECTYPE weight_dtype = UINT8;
    ARBPRECTYPE psum_dtype = UINT32;
    addr_t weight_step = sizeofArbPrecType(weight_dtype);
    addr_t ifmap_full_addr = args.ifmap_full_addr;
    int weight_load_latency = num_cols;
    size_t dsize = sizeofArbPrecType((ARBPRECTYPE)ifmap_dtype);
    //size_t tsize;
    addr_t weight_base_addr;
    LdWeightsArgs weight_args;
    MatMulArgs    matmul_args;
    PoolArgs      pool_args;

    weight_args.dtype = weight_dtype;
    weight_args.num_cols = num_cols;
    weight_args.num_rows = num_rows;
    weight_args.x_step = weight_step;
    weight_args.x_num = args.w_s;
    weight_args.y_step = weight_step * args.w_s;
    weight_args.y_num = args.w_r;
    weight_base_addr = args.filter_full_addr + (weight_load_latency-1) * (weight_args.y_num * weight_args.y_step);
    weight_args.weight_full_addr = weight_base_addr;

    size_t tile_x_dim = Constants::tile_size;
    size_t tile_y_dim = Constants::tile_size;
    matmul_args.ifmap_full_addr   = 0xdead;
    size_t x_tile_or_ofmap = 
        ofmap_cols > tile_x_dim ? tile_x_dim : ofmap_cols;
    matmul_args.x_step = 1;
    size_t y_tile_or_ofmap = 
        ofmap_rows > tile_y_dim ? tile_y_dim : ofmap_rows;
    matmul_args.y_step = ifmap_cols;
    matmul_args.dtype = ifmap_dtype;
    matmul_args.psum_dtype = psum_dtype;
    matmul_args.num_rows = num_rows;
    matmul_args.num_cols = num_cols;

    ARBPRECTYPE pool_dtype = UINT32;
    size_t psize = sizeofArbPrecType((ARBPRECTYPE)pool_dtype);
    size_t  tile_x = x_tile_or_ofmap;
    size_t  tile_y = y_tile_or_ofmap;
    size_t  tile_xx = ofmap_cols % tile_x_dim ? ofmap_cols % tile_x_dim : tile_x;
    size_t  tile_yy = ofmap_rows % tile_y_dim ? ofmap_rows % tile_y_dim : tile_y;
    pool_args.pool_func = IDENTITY_POOL;
    pool_args.dtype     = pool_dtype;
    pool_args.src_full_addr = psum_buffer_base;
    pool_args.src_x_step= 1;
    pool_args.src_y_step= 1;
    pool_args.src_z_step= 1;
 //   pool_args.src_x_num = tile_x * tile_y;
    pool_args.src_y_num = 1;
    pool_args.src_z_num = 1;
    pool_args.dst_x_step = 1;
 //   pool_args.dst_x_num = tile_x;
    pool_args.dst_y_step = ofmap_cols;
 //   pool_args.dst_y_num = tile_y;
    pool_args.dst_full_addr = args.ofmap_full_addr;
    pool_args.num_partitions = args.w_m;
    //tsize = tile_x * tile_y * psize;

    /* go through each weight in the filter, cannot combine r and s into one, 
     * because ifmap_full_addr calc needs to seperate them */
    /* go through each weight in the filter, cannot combine r and s into one, because ifmap_full_addr 
	   calc needs to seperate them */
    pool_args.dst_full_addr =args.ofmap_full_addr;
    unsigned int curr_tile = 0;
    unsigned int ii, jj;
    size_t t_x, t_y;
    for (unsigned int i = 0; i < tile_rows; i++) {
        t_y = (i == tile_rows - 1) ? tile_yy : tile_y;
        for (unsigned int j = 0; j < tile_cols; j++, curr_tile++) {
            t_x = (j == tile_cols - 1) ? tile_xx : tile_x;
            unsigned int curr_weight = 0;
            uop.PUSH(new DynamicInstruction<LdWeightsArgs>(weight_args));
            for (unsigned int r = 0; r <  filter_rows; r++) {
                for (unsigned int s = 0; s < filter_cols; s++, curr_weight++) {
                    /* go through each ofmap pixel this weight operates one*/
                    matmul_args.psum_start = (r == 0 && s == 0);
                    matmul_args.psum_stop = (curr_weight == 
                            (filter_rows * filter_cols - 1));
                    matmul_args.x_num = t_x;
                    matmul_args.y_num = t_y;
                    ii = i * Constants::tile_size;
                    jj = j * Constants::tile_size;
                    matmul_args.ifmap_full_addr = ifmap_full_addr +
                        ((r + ii)* ifmap_cols + (s + jj)) * dsize;
                    uop.PUSH(new DynamicInstruction<MatMulArgs>(matmul_args));
                    if (curr_weight < (filter_rows * filter_cols - 1)) {
                        weight_args.weight_full_addr += weight_step;
                        uop.PUSH(new DynamicInstruction<LdWeightsArgs>(weight_args));
                    } else {
                        weight_args.weight_full_addr = weight_base_addr;
                    }
                }
            }
            // coudl replace t_y with tile_y to overwrite sloppily
            pool_args.src_x_num = t_x * t_y;
            pool_args.dst_x_num = t_x;
            pool_args.dst_y_num = t_y;
            uop.PUSH(new DynamicInstruction<PoolArgs>(pool_args));
            if (j < (tile_cols - 1)) {
                pool_args.dst_full_addr += t_x * psize;
            } else {
                pool_args.dst_full_addr = args.ofmap_full_addr +
                    ((j+1-1) * (tile_x * tile_y) + t_x * t_y) *  psize;
                    //(i + 1) * (tile_cols-1)*(tile_x * tile_y) * psize +
                    //(tile_xx * tile_y) * psize;
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
    ARBPRECTYPE psum_dtype  = UINT32;
    ARBPRECTYPE ifmap_dtype = UINT8;
    int weight_load_latency = num_ofmaps;
    int curr_opixel, curr_weight;
    int weight_step   = sizeofArbPrecType(args.weight_dtype);
    assert(weight_load_latency < 64 && "Tiling not implemented yet, don't have enough columsn for # ofmaps!");

    printf("warning: bit rot ahead\n");
    /* signals that will stay constant for entire convolution */
    es.weight_dtype  = args.weight_dtype;
    es.psum_dtype    = psum_dtype;
    es.activation    = IDENTITY; 
    es.pool_type     = IDENTITY_POOL;
    es.pool_dtype    = psum_dtype;
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
                        ((r * ifmap_cols + s) + (e * ifmap_cols) + f) * sizeofArbPrecType((ARBPRECTYPE)ifmap_dtype);


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
