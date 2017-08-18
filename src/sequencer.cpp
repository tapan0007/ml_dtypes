#include "sequencer.h"
#include "types.h"
#include "string.h"

/*------------------------------------
 * EdgeSignalsInstruction
 *------------------------------------ */
void EdgeSignalsInstruction::dump(bool header)
{   
    if (header) {
        printf("rc cc |  iv   ia  |  wv wa   wd  wt wc | pi pd ps pe | av af | pv pt px py pd | oa \n");
    }
    printf("%2d %2d | %2d 0x%-3lx  | %2d 0x%-3lx  %2d %2d %2d | %2d %2d %2d %2d | %2d %2d | %2d %2d %2d %2d %2d | 0x%-3lx  \n",
            es.row_countdown, es.column_countdown, 
            es.ifmap_valid, es.ifmap_full_addr,
            es.weight_valid, es.weight_full_addr, es.weight_dtype, es.weight_toggle, es.weight_clamp,
            es.psum_id, es.psum_dtype, es.psum_start, es.psum_end,
            es.activation_valid, es.activation,
            es.pool_valid, es.pool_type, es.pool_dimx, es.pool_dimy, es.pool_dtype,
            es.ofmap_full_addr);

}
 
void
EdgeSignalsInstruction::execute(Sequencer *seq) {

    seq->es = es;
    seq->raw_signal = true;
}

/*------------------------------------
 * LdWeightsInstruction
 *------------------------------------ */
LdWeights::LdWeights(const LdWeightsArgs &_args) : args(_args) { }

LdWeights::~LdWeights() {}

void  LdWeights::execute(Sequencer *seq) {
    seq->es.weight_valid = true;
    seq->es.weight_dtype = args.weight_dtype;
    seq->es.weight_full_addr = args.weight_full_addr;
    seq->es.weight_clamp = (args.weight_columns == 1);
    seq->es.row_countdown = args.weight_rows; 
    seq->weight_columns = args.weight_columns;
    seq->weight_step = args.weight_step;
    seq->raw_signal = false;
}

/*------------------------------------
 * Convolve
 *------------------------------------ */
MatMul::MatMul(const MatMulArgs &_args) : args(_args) { }

MatMul::~MatMul() {}

void  MatMul::execute(Sequencer *seq) {
    uint8_t dtype_size = fmapdtype_to_bytes[args.ifmap_dtype];
    /* signals that will stay constant for entire convolution */
    seq->es.ifmap_valid   = true;
    seq->es.ifmap_full_addr    = args.ifmap_full_addr;
    seq->es.ofmap_full_addr    = args.ofmap_full_addr;
    seq->es.row_countdown = args.num_rows; 
    seq->es.column_countdown = args.num_cols;
    seq->es.psum_start = args.psum_start;
    seq->es.psum_end = args.psum_end;
    seq->es.psum_dtype = (ARBPRECTYPE)args.psum_dtype;
    seq->es.pool_valid = args.psum_end; // FIXME - temporary hack
    seq->es.pool_dtype = (ARBPRECTYPE)args.psum_dtype; // FIXME - temporary hack to get results
    seq->es.psum_id = 0; // tmp
    seq->es.weight_toggle = true;
    seq->ifmap_base = args.ifmap_full_addr;
    seq->ifmap_x_num = args.ifmap_x_num_elements;
    seq->ifmap_y_num = args.ifmap_y_num_elements;
    seq->ifmap_x_cnt = 0;
    seq->ifmap_y_cnt = 0;
    seq->ifmap_eol_stride = (args.ifmap_y_step - (args.ifmap_x_step * args.ifmap_x_num_elements)) * dtype_size;
    seq->raw_signal = false;

}


/*------------------------------------
 * Sequencer
 *------------------------------------ */

Sequencer::Sequencer() : es(), raw_signal(false), clock(0) {
}

Sequencer::~Sequencer() {
}

bool
Sequencer::synch() {
    return es.ifmap_valid  || es.weight_valid;
}

#define COND_SET(X, VAL) (X == VAL) ? false : X=VAL

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
    /* update state - feed pixel */
    if (es.ifmap_valid) {
        /* es state */
        if ((ifmap_x_cnt == ifmap_x_num - 1) && (ifmap_y_cnt == ifmap_y_num - 1)) {
            /* clear state */
            es.ifmap_valid = false;
            es.psum_start = false;
            es.psum_end = false;
            es.pool_valid = false;
            es.activation_valid = false;
        } else {
            es.ifmap_full_addr += fmapdtype_to_bytes[es.ifmap_dtype];
            if (++ifmap_x_cnt >= ifmap_x_num) {
                if (++ifmap_y_cnt < ifmap_y_num) {
                    es.ifmap_full_addr += ifmap_eol_stride;
                }
                ifmap_x_cnt = 0;
            }
            es.psum_id++;
            es.ofmap_full_addr += sizeofArbPrecType(es.psum_dtype);
        }
        COND_SET(es.weight_toggle, false);
    }
    /* update state - feed weight */
    if (es.weight_valid) {
        /* es state */
        if (--weight_columns) {
            es.weight_full_addr -= weight_step;
        }
        if (weight_columns == 1) {
            es.weight_clamp = true;
        } else if (weight_columns == 0) {
            assert(es.weight_clamp);
            es.weight_clamp = false;
            es.weight_valid = false;
        } 
    }

    clock++;
}


EdgeSignals
Sequencer::pull_edge() {
    return es;
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
    int filter_rows = args.w_r;
    int filter_cols = args.w_s;
    int ifmap_rows = args.i_h;
    int ifmap_cols = args.i_w;
    int ofmap_rows = ifmap_rows - filter_rows + 1;
    int ofmap_cols =  ifmap_cols - filter_cols + 1;
    int num_cols = args.w_m;
    int num_rows = args.i_c;
    ARBPRECTYPE ifmap_dtype = UINT8;
    ARBPRECTYPE weight_dtype = UINT8;
    ARBPRECTYPE psum_dtype = UINT32;
    addr_t weight_step = sizeofArbPrecType(weight_dtype);
    addr_t ifmap_full_addr = args.ifmap_full_addr;
    int weight_load_latency = num_cols;
    assert(weight_load_latency < 64 && "Tiling not implemented yet, too many ofmaps!");

    LdWeightsArgs weight_args;
    weight_args.weight_dtype = weight_dtype;
    weight_args.weight_columns = num_cols;
    weight_args.weight_rows = num_rows;
    weight_args.weight_step = weight_step;
    weight_args.weight_full_addr = args.filter_full_addr + (weight_load_latency-1) * weight_step;

    uop.PUSH(new LdWeights(weight_args));

    MatMulArgs matmul_args;
    matmul_args.ifmap_full_addr   = 0xdead;
    matmul_args.ifmap_x_num_elements = ofmap_cols;
    matmul_args.ifmap_x_step = 1;
    matmul_args.ifmap_y_num_elements = ofmap_rows;
    matmul_args.ifmap_y_step = ifmap_cols;
    matmul_args.ofmap_full_addr = args.ofmap_full_addr;
    matmul_args.ifmap_dtype = ifmap_dtype;
    matmul_args.psum_dtype = psum_dtype;
    matmul_args.num_rows = num_rows;
    matmul_args.num_cols = num_cols;

    /* go through each weight in the filter */
    int curr_weight = 0;
    for (int r = 0; r <  filter_rows; r++) {
        for (int s = 0; s < filter_cols; s++, curr_weight++) {
            /* go through each ofmap pixel this weight operates one*/
            matmul_args.psum_start = (r == 0 && s == 0);
            matmul_args.psum_end = (curr_weight == (filter_rows * filter_cols - 1));
            matmul_args.ifmap_full_addr = ifmap_full_addr + (r * ifmap_cols + s) * fmapdtype_to_bytes[ifmap_dtype];
            uop.PUSH(new MatMul(matmul_args));
            if (curr_weight < (filter_rows * filter_cols - 1)) {
                weight_args.weight_full_addr += weight_load_latency * weight_step;
                uop.PUSH(new LdWeights(weight_args));
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

    /* signals that will stay constant for entire convolution */
    es.weight_dtype  = args.weight_dtype;
    es.psum_dtype    = psum_dtype;
    es.activation    = IDENTITY; 
    es.pool_type     = NO_POOL;
    es.pool_dtype    = psum_dtype;
    es.row_countdown = ifmap_channels; 
    es.column_countdown = num_ofmaps;


    /* LOAD WEIGHTS: 
     * first loading of weights for all ofmaps, weight_clamp on last weight */
    es.weight_valid  = true;
    es.weight_full_addr = args.filter_full_addr + (weight_load_latency -1) * sizeofArbPrecType(es.weight_dtype);
    for (int i = 0; i < weight_load_latency - 1; i++) {
        uop.PUSH(new EdgeSignalsInstruction(es));
        es.weight_full_addr -= sizeofArbPrecType(es.weight_dtype);
    }
    es.weight_clamp = true;
    uop.PUSH(new EdgeSignalsInstruction(es));

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
                    es.psum_id  = curr_opixel;
                    es.psum_start = (r == 0) && (s == 0); // only clear psum buffer on first weight
                    es.weight_toggle = (e == 0) && (f == 0); // only toggle on first ofmap pixel

                    /* LOAD PIXEL */
                    es.ifmap_full_addr = args.ifmap_full_addr +
                        ((r * ifmap_cols + s) + (e * ifmap_cols) + f) * fmapdtype_to_bytes[ifmap_dtype];


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
                        es.psum_end   = true;
                        es.activation_valid = true;
                        es.pool_valid = true;
                        es.ofmap_full_addr = args.ofmap_full_addr + curr_opixel * 
                            sizeofArbPrecType(psum_dtype);
                    }
                    
                    uop.PUSH(new EdgeSignalsInstruction(es));
                }
            }
        }
    }

    dump();
}

bool
Sequencer::done() {
    return uop.empty() && !es.ifmap_valid && !es.weight_valid;
}
