#ifndef PE_ARRAY_H
#define PE_ARRAY_H

#include "pe.h"
#include <array>
#include <cstdint>

class ProcessingElementArray {
    public:
        ProcessingElementArray(int num_rows = 128, int num_cols = 64);
        ~ProcessingElementArray();
        void connect_west(int row, EWInterface *ew);
        void connect_north(int col, NSInterface *ns);
        void step();
        void dump(FILE *f);
    private:
        ProcessingElement **pe_array;
        ZeroInterfaceGenerator ns_generator;
        int num_rows;
        int num_cols;
};

#endif //PE_ARRAY_H
