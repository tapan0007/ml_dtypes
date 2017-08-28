#ifndef PSUM_BUFFER_H
#define PSUM_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

class PSumBuffer : public EdgeInterface, public PSumActivateInterface {
    public:
        PSumBuffer();
        ~PSumBuffer();
        EdgeSignals pull_edge();
        PSumActivateSignals pull_psum();
        void connect_west(EdgeInterface *);
        void connect_north(PeNSInterface *);
        void set_address(addr_t base_addr);
        void step();
    private:
        ArbPrecData              pool();
        ArbPrecData              activation(ArbPrecData pixel);
        char                    *ptr; // pointer to memory
        PeNSSignals              ns;
        EdgeSignals              ew;
        PeNSInterface            *north;
        EdgeInterface            *west;
        addr_t                    ready_addr;
        union {
            char      *char_ptr;
            uint32_t  *uint32_ptr;
            int32_t   *int32_ptr;
            uint64_t  *uint64_ptr;
            int64_t   *int64_ptr;
            float     *fp32_ptr;
        } __attribute((__packed__)) ptrs;

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
