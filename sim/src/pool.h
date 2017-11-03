#ifndef POOL_BUFFER_H
#define POOL_BUFFER_H

#include "sigint.h"
#include "io.h"
#include <vector>

class Pool : public PoolInterface {
    public:
        Pool(MemoryMap *_memory) : memory(_memory) {}
        PoolSignals pull_pool();
        void connect(PoolInterface *);
        void step();
    private:
        PoolSignals              ps = {0};
        PoolInterface           *connection = nullptr;
        ArbPrecData              pool_pixel = {0};
        addr_t src_partition_size = 0;
        addr_t dst_partition_size = 0;
        unsigned int             pool_cnt = 0;
        MemoryMap               *memory = nullptr;
};

class PoolArray {
    public:
        PoolArray(MemoryMap *mmap, size_t n_pools);
        ~PoolArray();
        Pool& operator[](int index);
        void connect(PoolInterface *);
        void step();
    private:
        std::vector<Pool>     pooler;

};
#endif
