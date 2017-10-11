#ifndef SIGINT_H
#define SIGINT_H
#include "uarch_cfg.h"
#include "tpb_isa.h"
#include "arb_prec.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
typedef struct PeEWSignals {
    ArbPrecData pixel;
    bool    pixel_valid;
    ArbPrecData weight;
    ARBPRECTYPE weight_dtype;
    bool    weight_toggle;
} PeEWSignals;

typedef struct PeNSSignals {
    ArbPrecData partial_sum;
} PeNSSignals;

typedef struct PSumActivateSignals {
    bool    valid;
    ArbPrecData partial_sum;
    ARBPRECTYPE dtype;
} PSumActivateSignals;

typedef struct ActivateSbSignals {
    bool    valid;
    ArbPrecData partial_sum;
    ARBPRECTYPE dtype;
} ActivateSbSignals;

#define ADDR_UNION(PREF)  \
    union { \
        struct { \
            uint64_t PREF##_addr : BANK_BITS; \
            uint64_t PREF##_bank : ADDRESS_BITS - BANK_BITS; \
        }; \
        uint64_t PREF##_full_addr : ADDRESS_BITS; \
    }; \


typedef struct EdgeSignals {
    uint8_t        row_valid;
    uint8_t        row_countdown;    // SB       | Which SB rows is this valid for?
    uint8_t        column_valid;
    uint8_t        column_countdown; // PSUM     | Which PSUM columns is this valid for?

    bool           pad_valid;
    bool           ifmap_valid;      // SB & PSUM| read and shift pixel from SB, PSUM should use result for MAC
    ADDR_UNION(ifmap);
    ARBPRECTYPE      ifmap_dtype;       // SB       | what type of pixel are we loading?

    bool           weight_valid;     // SB       | read and shift weight from SB
    ADDR_UNION(weight)      // SB       | weight address
    ARBPRECTYPE    weight_dtype;     // SB       | what type of weight are we loading?
    bool           weight_toggle;    // PE       | should the PE toggle the weight ptr bit? FIXME: could be id instead?
    bool           weight_clamp;     // PE       | broadcast signal to tell PEs in a row to "clamp" the weight passing through them

    ADDR_UNION(psum)      // SB       | weight address
    bool           psum_start;       // PSUM     | Clear psum buffer for new calc
    bool           psum_stop;         // PSUM     | Psum calc is done | FIXME : psum_start/end could be combined but we'd lose debugability

    bool           activation_valid; // PSUM     | Should we perform an activation on psum id?
    ACTIVATIONFUNCTION activation;   // PSUM     | Which activation func should we perform?
 
} EdgeSignals;

typedef struct PoolSignals {
    bool          valid;
    POOLFUNC      func;
    ARBPRECTYPE   dtype;
    ADDR_UNION(src);
    ADDR_UNION(dst);
    bool          start;
    bool          stop;
    uint8_t       countdown;
} PoolSignals;


// ----------------------------------------------------------
// Interfaces
// ----------------------------------------------------------
class Instruction;
class UopFeedInterface
{
    public:
        virtual  bool         empty() = 0;
        virtual  void        *front() = 0;
        virtual  void         pop() = 0;
};

class PeEWInterface
{
    public:
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
        virtual bool pull_clamp() = 0;
};

class PoolInterface
{
    public:
        PoolInterface() {};
        ~PoolInterface() {};
        virtual PoolSignals pull_pool() = 0;
};
// ----------------------------------------------------------
// Dummy Generators
// ----------------------------------------------------------
class ZeroPeNSGenerator : public PeNSInterface {
    public:
        PeNSSignals pull_ns() {return {0};}
};




#endif  
