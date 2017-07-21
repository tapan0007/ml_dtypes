#ifndef PE_ARRAY_H
#define PE_ARRAY_H

#include "pe.h"
#include "sequencer.h"
#include <array>
#include <cstdint>

class ProcessingElementArray {
    public:
        ProcessingElementArray(int num_rows = 128, int num_cols = 64);
        ~ProcessingElementArray();
        void connect_west(int row, PeEWInterface *ew);
        void connect_north(int col, PeNSInterface *ns);
        void connect_statebuffer(int row, SbEWBroadcastInterface *sequencer);
        void step();
        void dump(FILE *f);
    private:
        ProcessingElement **pe_array;
        ZeroPeNSGenerator ns_generator;
        int num_rows;
        int num_cols;
};

#endif //PE_ARRAY_H
