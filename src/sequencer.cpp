#include "sequencer.h"
#include "types.h"
#include "string.h"

Sequencer::Sequencer() : clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    if (uop.size()) {
        uop.pop();
    }
    clock++;
}


EdgeSignals
Sequencer::pull_edge() {
    if (uop.size()) {
        return uop.front();
    } 
    return EdgeSignals{};
}

void Sequencer::dump_es(const EdgeSignals &es, bool header)
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
Sequencer::dump() {
    EdgeSignals es = {};
    int num = uop.size();
    int header;
    for (int i = 0; i < num; i++) {
        es = uop.front();
        header = (i == 0);
        dump_es(es, header);
        uop.pop();
        uop.push(es);
    }
}

static ARBPRECTYPE weight_to_psum_dtype[NUM_ARBPRECTYPE] = {[UINT8]=UINT32, [UINT32]=UINT32, [FP32]=FP32};

#define PUSH push

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
Sequencer::convolve(const ConvolveArgs &args)
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
    int num_rows = 128;
    int num_cols = 64;
    int curr_opixel, curr_weight;

    assert(weight_load_latency < num_cols && "Tiling not implemented yet, too many ofmaps!");

    /* signals that will stay constant for entire convolution */
    es.ifmap_stride  = sizeofArbPrecType(ifmap_dtype) * ifmap_rows * ifmap_cols;
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
        uop.PUSH(es);
        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
    }
    es.weight_clamp = true;
    uop.PUSH(es);

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
                        ((r * ifmap_cols + s) + (e * ifmap_cols) + f) * 
                        sizeofArbPrecType(ifmap_dtype);


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
                        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
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
                    
                    uop.PUSH(es);
                }
            }
        }
    }

    /* drain out results*/
    memset(&es, 0, sizeof(es));
    for (int i = 0; i < num_rows + num_ofmaps; i++) {
        uop.PUSH(es);
    }
    dump();
}

int
Sequencer::steps_to_do() {
    return uop.size();
}
