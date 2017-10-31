#ifndef PSUM_BUFFER_H
#define PSUM_BUFFER_H

#include "sigint.h"
#include <vector>

class MemoryMap;
class MemoryMapInstance;

class PSumBuffer : public EdgeInterface {
    public:
        PSumBuffer(MemoryMap *mmap, addr_t base, size_t nbytes);
        EdgeSignals pull_edge();
        void connect_west(EdgeInterface *);
        void connect_north(PeNSInterface *);
        PSumActivateSignals pull_psum();
        void step();
    private:
        ArbPrecData              pool();
        PeNSSignals              ns;
        EdgeSignals              ew = {0};
        PeNSInterface            *north = nullptr;
        EdgeInterface            *west  = nullptr;
        MemoryMapInstance        *mem;

};

class PSumBufferArray {
    public:
        PSumBufferArray(MemoryMap *, addr_t base, size_t n_cols);
        ~PSumBufferArray();
        PSumBuffer& operator[](int index);
        void connect_west(EdgeInterface *);
        void connect_north(int col_id, PeNSInterface *);
        void step();
    private:
        std::vector<PSumBuffer>     col_buffer;

};
#endif 
