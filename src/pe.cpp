#include "pe.h"

ProcessingElement::ProcessingElement() : partial_sum(ArbPrec(uint32_t(0))), weight_id(0), weight{ArbPrec(uint8_t(0)), ArbPrec(uint8_t(0))}, north(NULL), west(NULL)
{
}

ProcessingElement::~ProcessingElement() {
}

PeNSSignals ProcessingElement::pull_ns() {
    return PeNSSignals(partial_sum);
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

void ProcessingElement::connect_sequencer(SequencerInterface *_sequencer) {
    sequencer = _sequencer;
}

void ProcessingElement::step() {
    PeEWSignals in_ew = west->pull_ew();
    PeNSSignals in_ns = north->pull_ns();
    bool in_clamp = sequencer->pull_clamp();
    
    if (in_clamp) {
        weight[!weight_id] = in_ew.weight;
    }
    if (in_ew.toggle_weight) {
        weight_id = !weight_id;
    }

    ew.pixel = in_ew.pixel;
    partial_sum = in_ns.partial_sum + in_ew.pixel * weight[weight_id];
}

void ProcessingElement::dump(FILE *f) {
    fprintf(f, "[p=");
    ew.pixel.dump(f);
    fprintf(f, ",w=");
    weight[weight_id].dump(f);
    fprintf(f, ",s=");
    partial_sum.dump(f);
    fprintf(f, "]");
}
