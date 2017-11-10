#ifndef PE_H
#define PE_H

#include "sigint.h"

class ProcessingElement : public PeEWInterface, public PeNSInterface  {
    public:
        /* make connections */
        ProcessingElement() 
        { ew.weight_toggle = false; ew.pixel_valid = false; }
        void connect_west(PeEWInterface *west);
        void connect_north(PeNSInterface *north);
        void connect_statebuffer(SbEWBroadcastInterface *sb);
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
        ArbPrecData partial_sum;
        /* caching of weights, fg and bg, with id ptr */
        int weight_id;
        ArbPrecData weight[2];
        ARBPRECTYPE weight_dtype[2];
        /* pointers to slave to get inputs from */
        PeNSInterface *north = nullptr;
        PeEWInterface *west  = nullptr;
        SbEWBroadcastInterface   *sb_west = nullptr;
};

#endif  //PE_H
