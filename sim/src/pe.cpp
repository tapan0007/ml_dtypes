#include "pe.h"

PeNSSignals ProcessingElement::pull_ns() {
    return PeNSSignals{partial_sum};
}

PeEWSignals ProcessingElement::pull_ew() {
    return ew;
}

void ProcessingElement::connect_west(PeEWInterface *_west) {
    west = _west;
}

void ProcessingElement::connect_north(PeNSInterface *_north) {
    north = _north;
}

void ProcessingElement::connect_statebuffer(SbEWBroadcastInterface *_sb) {
    sb_west = _sb;
}

void ProcessingElement::step() {
    PeEWSignals in_ew = west->pull_ew();
    PeNSSignals in_ns = north->pull_ns();
    bool in_clamp = sb_west->pull_clamp();
    ArbPrecData product;
    if (in_ew.weight_toggle) {
        weight_id = !weight_id;
    }
    if (in_clamp) {
        weight[!weight_id] = in_ew.weight;
        weight_dtype[!weight_id] = in_ew.weight_dtype;
        in_ew.weight.raw = 0;
        in_ew.weight_dtype = INVALID_ARBPRECTYPE;
    }

    if (in_ew.pixel_valid && weight_dtype[weight_id] != INVALID_ARBPRECTYPE) {
        ARBPRECTYPE out_dtype;
        partial_sum = ArbPrec::fma(in_ew.pixel, 
                weight[weight_id],
                in_ns.partial_sum,
                weight_dtype[weight_id], 
                out_dtype);
    } else {
        partial_sum = in_ns.partial_sum;
    }
    ew = in_ew;
}

void ProcessingElement::dump(FILE *f) {
    fprintf(f, "[p=");
    ArbPrec::dump(f, ew.pixel, weight_dtype[weight_id]);
    fprintf(f, ",w=");
    ArbPrec::dump(f, weight[weight_id], weight_dtype[weight_id]);
    fprintf(f, ",s=");
    ArbPrec::dump(f, partial_sum, UINT32);
    fprintf(f, "]");
}
