#include "pe.h"

ProcessingElement::ProcessingElement() :weight(ArbPrec(uint8_t(0))), partial_sum(ArbPrec(uint32_t(0)))
{
}

ProcessingElement::~ProcessingElement() {
}

NSSignals ProcessingElement::pull_ns() {
    return NSSignals(partial_sum);
}

EWSignals ProcessingElement::pull_ew() {
    return ew;
}

void ProcessingElement::connect(EWInterface *_west, NSInterface *_north) {
    west = _west;
    north = _north;
}

void ProcessingElement::step() {
    EWSignals in_ew = west->pull_ew();
    NSSignals in_ns = north->pull_ns();
    weight = in_ew.weight;
    ew.pixel = in_ew.pixel;
    partial_sum = in_ns.partial_sum + in_ew.pixel * weight;
}

void ProcessingElement::dump(FILE *f) {
    fprintf(f, "[p=");
    ew.pixel.dump(f);
    fprintf(f, ",w=");
    weight.dump(f);
    fprintf(f, ",s=");
    partial_sum.dump(f);
    fprintf(f, "]");
}
