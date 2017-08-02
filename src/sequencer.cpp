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
    ARBPRECTYPE psum_dtype  = weight_to_psum_dtype[args.weight_dtype];
    ARBPRECTYPE ifmap_dtype = UINT8;
    int pipe_depth = 128;

    es.weight_clamp = false;
    es.ifmap_valid = false;
    es.weight_valid = true;
    es.weight_addr = args.filter_addr + filter_stride - 1;
    es.weight_stride = filter_stride;
    es.weight_dtype = args.weight_dtype;
    es.weight_toggle = false;
    es.row_countdown = args.i_s;
    uop.PUSH_BACK(es);

    /* step in weights for all ofmaps, weight_clamp on last step */
    for (int i = 0; i < args.w_s; i++) {
        if (i == args.w_s - 1) {
            es.weight_clamp = true;
        }
        uop.PUSH_BACK(es);
        es.weight_addr -= sizeofArbPrecType(es.weight_dtype);
    }

    /* unweight_clamp, stop feeding weights, feed ifmaps instead */
    es.weight_clamp = false;
    es.ifmap_valid = true;
    es.ifmap_addr = args.ifmap_addr;
    es.ifmap_stride = sizeofArbPrecType(ifmap_dtype) * args.i_t * args.i_u;
    es.row_countdown = args.i_s;
    es.psum_start = true;
    es.psum_end = true;
    es.psum_id = 0; 
    es.ofmap_addr = args.ofmap_addr;
    es.ofmap_stride = sizeofArbPrecType(psum_dtype) * args.w_s * args.i_t * args.i_u;
    es.psum_dtype = psum_dtype;
    es.column_countdown = args.w_s;
    es.weight_valid = false;
    es.weight_toggle = true;
    es.activation_valid = true;
    es.activation_valid = IDENTITY;
    es.pool_valid = true;
    es.pool_type = NO_POOL;
    es.pool_dtype = psum_dtype;

    /* unweight_clamp, done toggling */
    for (int i = 0; i < args.i_t * args.i_u; i++) {
        uop.PUSH_BACK(es);
        if (i == 0) {
            es.weight_clamp = false;
            es.weight_toggle = false;
        }
        es.psum_id++;
        es.ifmap_addr += sizeofArbPrecType(ifmap_dtype);
        es.ofmap_addr += sizeofArbPrecType(psum_dtype);
    }
    /* drain out */
    es.psum_start = false;
    es.psum_end = false;
    es.ifmap_valid = false;
    es.weight_valid = false;
    for (int i = 0; i < pipe_depth + args.w_s; i++) {
        uop.PUSH_BACK(es);
    }
}

