#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include "types.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
//enum Opcode {GO=0, START_CALC, END_CALC, BUBBLE, NUM_OPCODE};
class PeEWSignals {
    public:
        PeEWSignals(ArbPrec _pixel = ArbPrec(), ArbPrec _weight = ArbPrec(), bool _toggle_weight = false)
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

class EdgeSignals {
    public:
        EdgeSignals(bool _ifmap_valid = false, bool _weight_valid = false, bool _toggle_weight = false, bool _clamp_weights = false, bool _start_psum = false, bool _end_psum = false, ArbPrecType _psum_dtype = UINT8, addr_t _psum_addr=MAX_ADDR) : 
            ifmap_valid(_ifmap_valid), weight_valid(_weight_valid), toggle_weight(_toggle_weight), clamp_weights(_clamp_weights), start_psum(_start_psum), end_psum(_end_psum), psum_dtype(_psum_dtype), psum_addr(_psum_addr) {}
        ~EdgeSignals() {}
        bool           ifmap_valid;
        bool           weight_valid;
        bool           toggle_weight;
        bool           clamp_weights;
        bool           start_psum;
        bool           end_psum;
        ArbPrecType    psum_dtype;
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

class EdgeInterface
{
    public:
        EdgeInterface() {};
        ~EdgeInterface() {};
        virtual EdgeSignals pull_edge() = 0;
};

class SbEWBroadcastInterface
{
    public:
        SbEWBroadcastInterface() {};
        ~SbEWBroadcastInterface() {};
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
