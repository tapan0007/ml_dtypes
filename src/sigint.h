#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include "types.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
//enum Opcode {GO=0, START_CALC, END_CALC, BUBBLE, NUM_OPCODE};
typedef struct PeEWSignals {
    ArbPrec pixel;
    ArbPrec weight;
    ARBPRECTYPE weight_dtype;
    bool    toggle_weight;
} PeEWSignals;

typedef struct PeNSSignals {
    ArbPrec partial_sum;
} PeNSSignals;

typedef struct PSumActivateSignals {
    bool    valid;
    ArbPrec partial_sum;
} PSumActivateSignals;

typedef struct ActivateSbSignals {
    bool    valid;
    ArbPrec partial_sum;
} ActivateSbSignals;



typedef struct EdgeSignals {
    uint8_t        row_countdown; // EDGE            | PSUM     |  >1 indicates addr_set valid 
    uint8_t        column_countdown; // EDGE          | PSUM     |  >1 indicates psum_end or addr_set valid 

    bool           ifmap_valid;   // EDGE                  | SB       | shift in a pixel?
    addr_t         ifmap_addr;
    addr_t         ifmap_stride;

    bool           weight_valid;  // EDGE                  | SB       | shift in a weight?
    addr_t         weight_addr;
    addr_t         weight_stride;
    ARBPRECTYPE    weight_dtype;    // EDGE &  EW-PE         | SB/PSUM/PE | data type of ifmap, weight, psum
    bool           toggle_weight; // EDGE &  EW-PE         | PE       | switch weight ptr
    bool           clamp_weights; // EDGE &  EW-PE(bcast)  | PE       | broadcast

    int            psum_id;
    ARBPRECTYPE    psum_dtype;
    bool           psum_start;    // EDGE                  | PSUM     | psum has started - clear psum # could get away with psum_clear, but harder to debug
    bool           psum_end;     // EDGE                  | PSUM       | psum is done, no more accumulations

    bool           activation_valid;
    ACTIVATIONFUNCTION activation;

    bool           pool_valid;
    POOLTYPE       pool_type;
    int            pool_dimx;
    int            pool_dimy;

    addr_t         ofmap_addr;
    addr_t         ofmap_stride;
} EdgeSignals;

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

class PSumActivateInterface
{
    public:
        PSumActivateInterface() {};
        ~PSumActivateInterface() {};
        virtual PSumActivateSignals pull_psum() = 0;
};

class ActivateSbInterface
{
    public:
        ActivateSbInterface() {};
        ~ActivateSbInterface() {};
        virtual ActivateSbSignals pull_activate() = 0;
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
        PeNSSignals pull_ns() {return PeNSSignals{ArbPrec((uint32_t)(0))};}
};


#endif  
