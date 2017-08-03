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

    int filter_stride = sizeofArbPrecType(args.weight_dtype) * args.w_s * args.w_t * args.w_u;
    int ofmap_rows = args.i_t - args.w_t + 1;
    int ofmap_cols = args.i_u - args.w_u + 1;
    ARBPRECTYPE psum_dtype  = weight_to_psum_dtype[args.weight_dtype];
    ARBPRECTYPE ifmap_dtype = UINT8;
    int num_rows = 128;

    es.weight_clamp = false;
    es.ifmap_valid = false;
    es.weight_valid = true;
    es.weight_addr = args.filter_addr + filter_stride - 1;
    es.weight_stride = filter_stride;
    es.weight_dtype = args.weight_dtype;
    es.weight_toggle = false;
    es.row_countdown = args.i_s;

    /* step in weights for all ofmaps, weight_clamp on last step */
    for (int i = 0; i < args.w_s; i++) {
        if (i == args.w_s - 1) {
            es.weight_clamp = true;
        }
        uop.PUSH_BACK(es);
        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
    }

    /* unweight_clamp, stop feeding weights, feed ifmaps instead */
    /* uncamp weight, stop sending weights, toggle weight  for first cycle */
    es.weight_clamp = false;
    es.weight_valid = false;
    es.weight_toggle = true;
    /* feed pixels */
    es.ifmap_valid = true;
    es.ifmap_addr = args.ifmap_addr;
    es.ifmap_stride = sizeofArbPrecType(ifmap_dtype) * args.i_t * args.i_u;
    es.row_countdown = args.i_s;
    /* 1x1 so we are done as soon as we start */
    es.psum_start = true;
    es.psum_end = true;
    es.psum_id = 0; 
    es.psum_dtype = psum_dtype;
    es.column_countdown = args.w_s;
    /* we are ready for activation too */
    es.activation_valid = true;
    es.activation_valid = IDENTITY;
    es.pool_valid = true;
    es.pool_type = NO_POOL;
    es.pool_dtype = psum_dtype;
    /* where results is going */
    es.ofmap_addr = args.ofmap_addr;
    es.ofmap_stride = sizeofArbPrecType(psum_dtype) * ofmap_rows  * ofmap_cols;

    /* push all pixels through systolic array */
    for (int i = 0; i < args.i_t * args.i_u; i++, es.psum_id=(es.psum_id + 1) % num_rows) {
        uop.PUSH_BACK(es);
        /* unweight_clamp, done toggling */
        if (i == 0) {
            es.weight_toggle = false;
        }
        es.ifmap_addr += sizeofArbPrecType(ifmap_dtype);
        es.ofmap_addr += sizeofArbPrecType(psum_dtype);
    }

    /* drain out results*/
    es.psum_start = false;
    es.psum_end = false;
    es.ifmap_valid = false;
    es.weight_valid = false;
    es.pool_valid = false;
    es.activation_valid = false;
    for (int i = 0; i < num_rows + args.w_s; i++) {
        uop.PUSH_BACK(es);
    }
}

int
Sequencer::steps_to_do() {
    return uop.size();
}
