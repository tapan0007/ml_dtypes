#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include <cstdint>

// ----------------------------------------------------------
// Signals
// ----------------------------------------------------------
class EWSignals {
    public:
        EWSignals(ArbPrec _pixel = ArbPrec(uint8_t(0)), ArbPrec _weight = ArbPrec(uint8_t(0))):pixel(_pixel), weight(_weight) {}
        ~EWSignals() {};
        ArbPrec pixel;
        ArbPrec weight;
};

class NSSignals {
    public:
        NSSignals(ArbPrec _partial_sum=ArbPrec(uint32_t(0))) : partial_sum(_partial_sum) {}
        ~NSSignals() {}
        ArbPrec partial_sum;
};


// ----------------------------------------------------------
// Interfaces
// ----------------------------------------------------------
class EWInterface
{
    public:
        EWInterface() {};
        ~EWInterface() {};
        virtual EWSignals pull_ew() = 0;
};

class NSInterface
{
    public:
        NSInterface() {};
        ~NSInterface() {};
        virtual NSSignals pull_ns() = 0;
};

class StateBufferShiftInterface
{
    public:
    StateBufferShiftInterface() {};
    ~StateBufferShiftInterface() {};
    virtual bool pull_shift() = 0;
};


// ----------------------------------------------------------
// Dummy Generators
// ----------------------------------------------------------
class RandomInterfaceGenerator : public EWInterface, public NSInterface {
    public:
        EWSignals pull_ew() {return EWSignals(ArbPrec((uint8_t)(rand() % 0xff)), ArbPrec(uint8_t(0))); };
        NSSignals pull_ns() {return NSSignals(ArbPrec((uint32_t)(0)));}
};


class ZeroInterfaceGenerator : public EWInterface, public NSInterface {
    public:
        EWSignals pull_ew() {return EWSignals(ArbPrec((uint8_t)(0)), ArbPrec(uint8_t(0))); };
        NSSignals pull_ns() {return NSSignals(ArbPrec((uint32_t)(0)));}
};


#endif  
