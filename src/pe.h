#ifndef PE_H
#define PE_H

#include "sigint.h"

class ProcessingElement : public EWInterface, public NSInterface  {
    public:
        ProcessingElement();
        ~ProcessingElement();
        void step();
        void dump(FILE *f);
        void connect_west(EWInterface *west);
        void connect_north(NSInterface *north);
        NSSignals pull_ns();
        EWSignals pull_ew();
    private:
        EWSignals ew;
        ArbPrec weight;
        ArbPrec partial_sum;
        // fixme - would be nice to make these const - 
        // but must be initialized in constructor?
        NSInterface *north;
        EWInterface *west;
};

#endif  //PE_H
