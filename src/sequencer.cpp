#include "sequencer.h"
#include "types.h"
#include "string.h"

/*------------------------------------
 * EdgeSignalsInstruction
 *------------------------------------ */
void EdgeSignalsInstruction::dump(bool header)
{   
    if (header) {
        printf("rc cc |  iv   ia   is  |  wv wa   ws   wd  wt wc | pi pd ps pe | av af | pv pt px py ps pd | oa os\n");
    }
    printf("%2d %2d | %2d 0x%-3lx 0x%-3lx | %2d 0x%-3lx 0x%-3lx %2d %2d %2d | %2d %2d %2d %2d | %2d %2d | %2d %2d %2d %2d %2d %2d | 0x%-3lx 0x%-3lx \n",
            es.row_countdown, es.column_countdown, 
            es.ifmap_valid, es.ifmap_addr, es.ifmap_stride,
            es.weight_valid, es.weight_addr, es.weight_stride, es.weight_dtype, es.weight_toggle, es.weight_clamp,
            es.psum_id, es.psum_dtype, es.psum_start, es.psum_end,
            es.activation_valid, es.activation,
            es.pool_valid, es.pool_type, es.pool_dimx, es.pool_dimy, es.pool_stride, es.pool_dtype,
            es.ofmap_addr, es.ofmap_stride);

}
 
void
EdgeSignalsInstruction::execute(Sequencer *seq) {

    seq->es = es;
}

/*------------------------------------
 * LdWeightsInstruction
 *------------------------------------ */
LdWeights::LdWeights(const LdWeightsArgs &_args) : args(_args) { }

LdWeights::~LdWeights() {}

void  LdWeights::execute(Sequencer *seq) {
    seq->es.weight_valid = true;
    seq->es.weight_stride = args.weight_stride;
    seq->es.weight_dtype = args.weight_dtype;
    seq->es.weight_addr = args.weight_addr;
    seq->weight_num  = args.weight_num;
    seq->weight_step = args.weight_step;
}

/*------------------------------------
 * Convolve
 *------------------------------------ */
MatMul::MatMul(const MatMulArgs &_args) : args(_args) { }

MatMul::~MatMul() {}

void  MatMul::execute(Sequencer *seq) {

    static int ready = 1024;
    if (!ready) {
        ready = 1024;
        return;
    }
    ready--;
    /* signals that will stay constant for entire convolution */
    seq->es.ifmap_valid   = true;
    seq->es.ifmap_addr    = args.ifmap_addr;
    seq->es.ifmap_stride  = args.ifmap_stride;
    seq->es.ofmap_addr    = args.ofmap_addr;
    seq->es.ofmap_stride  = args.ofmap_stride;
    seq->es.psum_dtype    = args.psum_dtype;
    seq->es.row_countdown = args.num_ifmaps; 
    seq->es.column_countdown = args.num_ofmaps;
    seq->es.psum_start = args.psum_start;
    seq->es.psum_id = 0; // tmp
    seq->es.weight_toggle = true;
    seq->ifmap_step = args.ifmap_step;
    seq->ifmap_x_num = args.ifmap_width/args.ifmap_step;
    seq->ifmap_y_num = args.ifmap_height;
    seq->ifmap_x_cnt = 0;
    seq->ifmap_y_cnt = 0;
    seq->ofmap_step    = args.ofmap_step;

}


/*------------------------------------
 * Sequencer
 *------------------------------------ */

Sequencer::Sequencer() : es(), clock(0) {
}

Sequencer::~Sequencer() {
}


void
Sequencer::step() {
    /* empty the instruction queue */
    if (!uop.empty()) {
        Instruction *inst = uop.front();
        inst->execute(this);
        uop.pop();
        free(inst);
    }

    /* update state - feed pixel */
    if (es.ifmap_valid) {
        /* es state */
        if (ifmap_x_cnt < ifmap_x_num) {
            es.ifmap_addr += ifmap_step;
            ifmap_x_cnt++;
        } else {
            if (ifmap_y_cnt < ifmap_y_num) {
                es.ifmap_addr += ifmap_y_num * es.ifmap_stride;
                ifmap_y_cnt++;
            }
        }
        if (es.weight_toggle) {
           es.weight_toggle = false;
        }
        if (es.psum_start) {
            es.psum_start = false;
        }
        if (es.psum_end) {
            es.psum_end = false;
        }
        es.psum_id++;
        es.ofmap_addr += ofmap_step;
    }
    /* update state - feed weight */
    if (es.weight_valid) {
        /* es state */
        es.weight_addr -= weight_step;
        if (weight_num == 1) {
            es.weight_clamp = true;
            weight_num--;
        } else if (weight_num == 0) {
            es.weight_clamp = false;
            es.weight_valid = false;
        } else {
            weight_num--;
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

static ARBPRECTYPE weight_to_psum_dtype[NUM_ARBPRECTYPE] = {[UINT8]=UINT32, [UINT32]=UINT32, [FP32]=FP32};

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
    int num_ofmaps = args.w_m;
    int num_ifmaps = args.i_c;
    int weight_stride = sizeofArbPrecType(args.weight_dtype) * num_ofmaps * filter_rows * filter_cols;
    ARBPRECTYPE psum_dtype  = weight_to_psum_dtype[args.weight_dtype];
    ARBPRECTYPE ifmap_dtype = UINT8;
    ARBPRECTYPE weight_dtype = UINT8;
    addr_t weight_step = sizeofArbPrecType(weight_dtype);
    addr_t ifmap_step = sizeofArbPrecType(ifmap_dtype);
    addr_t ofmap_step = sizeofArbPrecType(psum_dtype);
    addr_t ifmap_stride  = ifmap_step * ifmap_rows * ifmap_cols;
    addr_t ofmap_stride  = ofmap_step * ofmap_rows * ofmap_cols;
    addr_t ifmap_addr = args.ifmap_addr;
    int weight_load_latency = num_ofmaps;
    assert(weight_load_latency < 64 && "Tiling not implemented yet, too many ofmaps!");

    LdWeightsArgs weight_args;
    weight_args.weight_dtype = UINT8;
    weight_args.weight_num = num_ofmaps;
    weight_args.weight_step = weight_step;
    weight_args.weight_stride = weight_stride;
    weight_args.weight_addr = args.filter_addr + (weight_load_latency -1) * weight_step;

    uop.PUSH(new LdWeights(weight_args));

    MatMulArgs matmul_args;
    matmul_args.ifmap_addr   = ifmap_addr,
    matmul_args.ifmap_stride = ifmap_stride;
    matmul_args.ifmap_step   = ifmap_step;
    matmul_args.ifmap_width = ifmap_cols;
    matmul_args.ifmap_height = ifmap_rows;
    matmul_args.ifmap_dtype = ifmap_dtype;
    matmul_args.ofmap_step = ofmap_step;
    matmul_args.ofmap_stride = ofmap_stride;
    matmul_args.psum_dtype = UINT32;
    matmul_args.num_ifmaps = num_ifmaps;
    matmul_args.num_ofmaps = num_ofmaps;
    matmul_args.psum_start = true;


    /* go through each weight in the filter */
    int curr_weight = 0;
    for (int r = 0; r <  filter_rows; r++) {
        for (int s = 0; s < filter_cols; s++, curr_weight++) {
            curr_weight = r * filter_cols + s;
            /* go through each ofmap pixel this weight operates one*/
            matmul_args.ifmap_addr += r * ifmap_cols + s;
            weight_args.weight_addr += weight_load_latency * weight_step;
            uop.PUSH(new MatMul(matmul_args));
            uop.PUSH(new LdWeights(weight_args));
            matmul_args.psum_start = false;
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
    int filter_stride = sizeofArbPrecType(args.weight_dtype) * num_ofmaps * filter_rows * filter_cols;
    ARBPRECTYPE psum_dtype  = weight_to_psum_dtype[args.weight_dtype];
    ARBPRECTYPE ifmap_dtype = UINT8;
    int weight_load_latency = num_ofmaps;
    int curr_opixel, curr_weight;
    int weight_step   = sizeofArbPrecType(args.weight_dtype);
    assert(weight_load_latency < 64 && "Tiling not implemented yet, don't have enough columsn for # ofmaps!");

    /* signals that will stay constant for entire convolution */
    es.ifmap_stride  = sizeofArbPrecType(ifmap_dtype) * ifmap_rows * ifmap_cols;
    es.ifmap_step =  sizeofArbPrecType(ifmap_dtype);
    es.weight_stride = filter_stride;
    es.ofmap_stride  = sizeofArbPrecType(psum_dtype) * ofmap_rows  * ofmap_cols;
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
    es.weight_addr = args.filter_addr + (weight_load_latency -1) * sizeofArbPrecType(es.weight_dtype);
    for (int i = 0; i < weight_load_latency - 1; i++) {
        uop.PUSH(new EdgeSignalsInstruction(es));
        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
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
                    es.ifmap_addr = args.ifmap_addr +
                        ((r * ifmap_cols + s) + (e * ifmap_cols) + f) * es.ifmap_step;


                    /* LOAD WEIGHTS */
                    if (curr_opixel == 0) {
                        /* we are calculating the weight addr of the next set
                         * (+1) and then trying to get to the last weight (+1)
                         * so we can load the weights in reverse */
                        es.weight_addr = args.filter_addr + (curr_weight + 1 + 1) * weight_load_latency 
                            * sizeofArbPrecType(es.weight_dtype) - 1;
                        es.weight_valid = true;
                    } else if (curr_opixel < weight_load_latency) {
                        assert(es.weight_valid);
                        es.weight_addr -= weight_step;
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
                        es.ofmap_addr = args.ofmap_addr + curr_opixel * 
                            sizeofArbPrecType(psum_dtype);
                    }
                    
                    uop.PUSH(new EdgeSignalsInstruction(es));
                }
            }
        }
    }

    /* drain out results*/
    memset(&es, 0, sizeof(es));
    for (int i = 0; i <= 128 + num_ofmaps; i++) {
        uop.PUSH(new EdgeSignalsInstruction(es));
    }
    dump();
}

bool
Sequencer::done() {
    return uop.empty();
}
