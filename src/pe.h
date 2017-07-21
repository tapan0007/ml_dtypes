#ifndef PE_H
#define PE_H

#include "sigint.h"

class ProcessingElement : public PeEWInterface, public PeNSInterface  {
    public:
        ProcessingElement();
        ~ProcessingElement();
        /* make connections */
        void connect_west(PeEWInterface *west);
        void connect_north(PeNSInterface *north);
        void connect_sequencer(SequencerInterface *sequencer);
        /* slave interface */
        PeNSSignals pull_ns();
        PeEWSignals pull_ew();
        /* debug */
        void dump(FILE *f);
        /* step! */
        void step();
    private:
        /* caching of signals passing through */
        PeEWSignals ew;
        /* caching of  data */
        ArbPrec partial_sum;
        /* caching of weights, fg and bg, with id ptr */
        int weight_id;
        ArbPrec weight[2];
        /* pointers to slave to get inputs from */
        PeNSInterface *north;
        PeEWInterface *west;
        SequencerInterface   *sequencer;
};

#endif  //PE_H
