#ifndef PSUM_BUFFER_H
#define PSUM_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

class PSumBuffer : public EdgeInterface {
    public:
        PSumBuffer(int n_entries = 128);
        ~PSumBuffer();
        EdgeSignals pull_edge();
        void connect_west(EdgeInterface *);
        void connect_north(PeNSInterface *);
        void step();
    private:
        PeNSSignals              ns;
        EdgeSignals              ew;
        PeNSInterface            *north;
        EdgeInterface            *west;
        std::vector<ArbPrec>     partial_sum;

};

class PSumBufferArray {
    public:
        PSumBufferArray(int n_cols = 64);
        ~PSumBufferArray();
        void connect_west(EdgeInterface *);
        void connect_north(int col_id, PeNSInterface *);
        void step();
    private:
        std::vector<PSumBuffer>     col_buffer;

};
#endif 
