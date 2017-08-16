#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include "types.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
typedef struct PeEWSignals {
    ArbPrec pixel;
    ArbPrec weight;
    ARBPRECTYPE weight_dtype;
    bool    weight_toggle;
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
                                     // CUSTOMER |  DESCRIPTION
    uint8_t        row_countdown;    // SB       | Which SB rows is this valid for?
    uint8_t        column_countdown; // PSUM     | Which PSUM columns is this valid for?

    bool           ifmap_valid;      // SB & PSUM| read and shift pixel from SB, PSUM should use result for MAC
    addr_t         ifmap_addr;       // SB       | pixel address
    addr_t         ifmap_step;       // SB       | what type of pixel are we loading?

    bool           weight_valid;     // SB       | read and shift weight from SB
    addr_t         weight_addr;      // SB       | weight address
    addr_t         weight_step;      // SB       | bytes between weights
    ARBPRECTYPE    weight_dtype;     // SB       | what type of weight are we loading?
    bool           weight_toggle;    // PE       | should the PE toggle the weight ptr bit? FIXME: could be id instead?
    bool           weight_clamp;     // PE       | broadcast signal to tell PEs in a row to "clamp" the weight passing through them

    int            psum_id;          // PSUM     | Which psum buffer in a given column is this result destined for?
    ARBPRECTYPE    psum_dtype;       // PSUM     | Dtype for psum ops FIXME: semi-redundant, could be inferred from weight_dtype
    bool           psum_start;       // PSUM     | Clear psum buffer for new calc
    bool           psum_end;         // PSUM     | Psum calc is done | FIXME : psum_start/end could be combined but we'd lose debugability

    bool           activation_valid; // PSUM     | Should we perform an activation on psum id?
    ACTIVATIONFUNCTION activation;   // PSUM     | Which activation func should we perform?
 
    bool           pool_valid;       // PSUM     | Should we perform a pool on psum id? 
    POOLTYPE       pool_type;        // PSUM     | Which  pooling func should we perform?
    int            pool_dimx;        // PSUM     | rows in pooling
    int            pool_dimy;        // PSUM     | cols in pooling
    ARBPRECTYPE    pool_dtype;       // PSUM     | What type of data is our pooling operating on? FIXME: semi-redundant? inferrable from weight_dtype

    addr_t         ofmap_addr;       // PSUM     | Dest address for pooling/activation result
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
