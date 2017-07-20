#ifndef SIGINT_H
#define SIGINT_H
#include "arb_prec.h"
#include <cstdint>

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


#endif  
