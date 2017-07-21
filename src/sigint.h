#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include "types.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
enum Opcode {GO=0, START_CALC, END_CALC, BUBBLE, NUM_OPCODE};
class PeEWSignals {
    public:
        PeEWSignals(ArbPrec _pixel = ArbPrec(uint8_t(0)), ArbPrec _weight = ArbPrec(uint8_t(0)), bool _toggle_weight = false)
            : pixel(_pixel), weight(_weight), toggle_weight(_toggle_weight) {}
        ~PeEWSignals() {};
        ArbPrec pixel;
        ArbPrec weight;
        bool    toggle_weight;
};

class PeNSSignals {
    public:
        PeNSSignals(ArbPrec _partial_sum=ArbPrec(uint32_t(0))) : partial_sum(_partial_sum) {}
        ~PeNSSignals() {}
        ArbPrec partial_sum;
};

class SbNSSignals {
    public:
        SbNSSignals(Opcode _op=BUBBLE, addr_t _psum_addr=MAX_ADDR) : op(_op), psum_addr(_psum_addr) {}
        ~SbNSSignals() {}
        Opcode         op;
        addr_t         psum_addr;
};

// ----------------------------------------------------------
// Interfaces
// ----------------------------------------------------------
class PeEWInterface
{
    public:
        PeEWInterface() {};
        ~PeEWInterface() {};
        virtual PeEWSignals pull_ew() = 0;
};

class PeNSInterface
{
    public:
        PeNSInterface() {};
        ~PeNSInterface() {};
        virtual PeNSSignals pull_ns() = 0;
};

class SbNSInterface
{
    public:
        SbNSInterface() {};
        ~SbNSInterface() {};
        virtual SbNSSignals pull_ns() = 0;
};

class SequencerInterface
{
    public:
        SequencerInterface() {};
        ~SequencerInterface() {};
        virtual bool pull_clamp() = 0;
};


// ----------------------------------------------------------
// Dummy Generators
// ----------------------------------------------------------
class ZeroPeNSGenerator : public PeNSInterface {
    public:
        PeNSSignals pull_ns() {return PeNSSignals(ArbPrec((uint32_t)(0)));}
};


#endif  
