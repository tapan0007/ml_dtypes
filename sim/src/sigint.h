#ifndef SIGINT_H
#define SIGINT_H
#include "uarch_cfg.h"
#include "tpb_isa.h"
#include "arb_prec.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
struct PeEWSignals {
    ArbPrecData pixel;
    bool    pixel_valid;
    ArbPrecData weight;
    ARBPRECTYPE weight_dtype;
    bool    weight_toggle;
};

struct PeNSSignals {
    ArbPrecData partial_sum;
};

struct PSumActivateSignals {
    bool    valid;
    ArbPrecData partial_sum;
    ARBPRECTYPE dtype;
};

struct ActivateSbSignals {
    bool    valid;
    ArbPrecData partial_sum;
    ARBPRECTYPE dtype;
};


struct Addr {
    union {
        uint64_t row :  ROW_BYTE_OFFSET_BITS + BANKS_PER_ROW_BITS +
            ROWS_PER_BANK_PER_ROW_BITS;
        uint64_t column : 1 + COLUMN_BYTE_OFFSET_BITS + BANKS_PER_COLUMN_BITS;
        uint64_t sys : ADDRESS_BITS;
    };
};


typedef struct EdgeSignals {
    uint8_t        row_valid;
    uint8_t        row_countdown; 
    uint8_t        column_valid;
    uint8_t        column_countdown; 

    bool           pad_valid;
    bool           ifmap_valid;     
    Addr           ifmap_addr;
    ARBPRECTYPE    ifmap_dtype;  

    bool           weight_valid;  
    Addr           weight_addr;
    ARBPRECTYPE    weight_dtype;
    bool           weight_toggle;
    bool           weight_clamp; 


    Addr           psum_addr;
    bool           psum_start;
    bool           psum_stop; 

    bool           activation_valid; 
    ACTIVATIONFUNCTION activation;  
 
} EdgeSignals;

typedef struct PoolSignals {
    bool          valid;
    POOLFUNC      func;
    ARBPRECTYPE   dtype;
    Addr          src_addr;
    Addr          dst_addr;
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
        PeNSSignals pull_ns() {return PeNSSignals();}
};




#endif  
