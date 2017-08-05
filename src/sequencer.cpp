#include "sequencer.h"
#include "types.h"

Sequencer::Sequencer() : clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    if (uop.size()) {
        uop.pop_front();
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

static ARBPRECTYPE weight_to_psum_dtype[NUM_ARBPRECTYPE] = {[UINT8]=UINT32, [UINT32]=UINT32, [FP32]=FP32};

#define PUSH_BACK push_back
void 
Sequencer::convolve(const ConvolveArgs &args)
{
    EdgeSignals es = {};

    int filter_rows = args.w_t;
    int filter_cols = args.w_u;
    int ifmap_rows = args.i_t;
    int ifmap_cols = args.i_u;
    int ofmap_rows = ifmap_rows - filter_rows + 1;
    int ofmap_cols =  ifmap_cols - filter_cols + 1;
    int num_ofmaps = args.w_s;
    int ifmap_channels = args.i_s;
    int filter_stride = sizeofArbPrecType(args.weight_dtype) * num_ofmaps * filter_rows * filter_cols;
    ARBPRECTYPE psum_dtype  = weight_to_psum_dtype[args.weight_dtype];
    ARBPRECTYPE ifmap_dtype = UINT8;
    int num_rows = 128;
    int weight_load_period = num_ofmaps >= 64 ? 64 : num_ofmaps;
    //int weight_reuse = ofmap_rows * ofmap_cols; // depends on psum buffers available, too

    es.weight_clamp = false;
    es.ifmap_valid = false;
    es.weight_valid = true;
    es.weight_stride = filter_stride;
    es.weight_dtype = args.weight_dtype;
    es.weight_toggle = false;
    es.row_countdown = ifmap_channels;

    /* step in weights for all ofmaps, weight_clamp on last step */
    es.weight_addr = args.filter_addr + (weight_load_period -1) * sizeofArbPrecType(es.weight_dtype);
    for (int i = 0; i < weight_load_period; i++) {
        if (i == weight_load_period - 1) {
            es.weight_clamp = true;
        }
        uop.PUSH_BACK(es);
        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
    }

    // dont with clamping/weights
    es.weight_valid = false;
    int weight_load_time = ofmap_rows * ofmap_cols - weight_load_period;
    /* for each pixel this weight will operate on */
    /* on first weight, so first ofmap, start psum */
    es.psum_start = true;
    es.psum_dtype = psum_dtype;
    for (int i = 0; i <  ifmap_rows - ofmap_rows + 1; i++) {
        for (int j = 0; j <  ifmap_cols - ofmap_cols + 1; j++) {
            es.psum_id = 0; // we are starting a new weight
            es.weight_toggle = true;
            es.weight_clamp = false;
            es.weight_valid = false;
            for (int r = 0; r < ofmap_rows; r++) {
                for (int s = 0; s < ofmap_cols; s++) {
                    /* unweight_clamp, stop feeding weights, feed ifmaps instead */
                    /* uncamp weight, stop sending weights, toggle weight  for first cycle */
                    /* feed pixels */
                    es.ifmap_valid = true;
                    // ifmap_addr + area base + offset to row in area + offset to col in area
                    es.ifmap_addr = args.ifmap_addr + ((i * ifmap_cols + j) + (r * ofmap_cols) + s) *sizeofArbPrecType(ifmap_dtype);
                    es.ifmap_stride = sizeofArbPrecType(ifmap_dtype) * ifmap_rows * ifmap_cols;
                    es.row_countdown = ifmap_channels;
                    es.column_countdown = num_ofmaps;
                    if ((i == ifmap_rows - ofmap_rows) && 
                            (j == ifmap_cols - ofmap_cols)) {/* on last weight, so last ofmap, end psum */
                        es.psum_end = true;
                        /* we are ready for activation too */
                        es.activation_valid = true;
                        es.activation_valid = IDENTITY;
                        es.pool_valid = true;
                        es.pool_type = NO_POOL;
                        es.pool_dtype = psum_dtype;
                        /* where results is going */
                        es.ofmap_stride = sizeofArbPrecType(psum_dtype) * ofmap_rows  * ofmap_cols;
                        es.ofmap_addr = args.ofmap_addr + (r * ofmap_cols + s) *sizeofArbPrecType(psum_dtype);
                    }
                    if (r * s == weight_load_time) {
                        // fixme, adding +1 twice? why? one to calculate the
                        // next one up, one because we want to calculate next
                        // one.
                        es.weight_addr = args.filter_addr + (i * ifmap_cols + j + 1 + 1) * weight_load_period * sizeofArbPrecType(es.weight_dtype) - 1;
                        es.weight_valid = true;
                    }
                    if (r * s == ofmap_rows * ofmap_cols - 1) {
                        es.weight_clamp = true;
                    } 
                    uop.PUSH_BACK(es);
                    if (es.weight_valid) {
                        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
                    }
                    es.weight_toggle = false;
                    es.psum_id++;
                }
            }
            es.psum_start = false; // no longer on first weight
        }
    }

    /* drain out results*/
    es.psum_start = false;
    es.psum_end = false;
    es.ifmap_valid = false;
    es.weight_valid = false;
    es.pool_valid = false;
    es.activation_valid = false;
    for (int i = 0; i < num_rows + num_ofmaps; i++) {
        uop.PUSH_BACK(es);
    }
}

int
Sequencer::steps_to_do() {
    return uop.size();
}
