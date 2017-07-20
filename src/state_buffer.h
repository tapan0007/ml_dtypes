#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include <array>
#include <cstdint>

class RandomInterfaces : public EWInterface, public NSInterface {
    public:
        EWSignals pull_ew() {return EWSignals(ArbPrec((uint8_t)(rand() % 0xff)), ArbPrec(uint8_t(0))); };
        NSSignals pull_ns() {return NSSignals(ArbPrec((uint32_t)(0)));}
};


#endif 
