#ifndef PE_ARRAY_H
#define PE_ARRAY_H

#include "pe.h"
#include <array>
#include <cstdint>

class Sequencer;

class ProcessingElementArray {
    public:
        ProcessingElementArray(int num_rows = 128, int num_cols = 64);
        ~ProcessingElementArray();
        void connect_west(int row, PeEWInterface *ew);
        void connect_north(int col, PeNSInterface *ns);
        void connect_statebuffer(int row, SbEWBroadcastInterface *sequencer);
        void step();
        void dump(FILE *f);
        int  num_rows();
        int  num_cols();
        ProcessingElement*& operator[](int index);
    private:
        ProcessingElement **pe_array;
        ZeroPeNSGenerator ns_generator;
        int n_rows;
        int n_cols;
};

#endif //PE_ARRAY_H
