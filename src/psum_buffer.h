#ifndef PSUM_BUFFER_H
#define PSUM_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

typedef struct PSumBufferEntry {
    ArbPrec partial_sum;
    bool    valid;
} PSumBufferEntry;

class PSumBuffer : public EdgeInterface, public PSumActivateInterface {
    public:
        PSumBuffer(int n_entries = 1024);
        ~PSumBuffer();
        EdgeSignals pull_edge();
        PSumActivateSignals pull_psum();
        void connect_west(EdgeInterface *);
        void connect_north(PeNSInterface *);
        void step();
    private:
        ArbPrec                  pool();
        ArbPrec                  activation(ArbPrec pixel);
        PeNSSignals              ns;
        EdgeSignals              ew;
        PeNSInterface            *north;
        EdgeInterface            *west;
        std::vector<PSumBufferEntry>     entry;
        int                      ready_id;

};

class PSumBufferArray {
    public:
        PSumBufferArray(int n_cols = 64);
        ~PSumBufferArray();
        PSumBuffer& operator[](int index);
        void connect_west(EdgeInterface *);
        void connect_north(int col_id, PeNSInterface *);
        void step();
    private:
        std::vector<PSumBuffer>     col_buffer;

};
#endif 
