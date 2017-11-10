#ifndef POOL_BUFFER_H
#define POOL_BUFFER_H

#include "sigint.h"
#include "io.h"
#include <vector>

class Pool : public PoolInterface {
    public:
        Pool(MemoryMap *_memory) : memory(_memory) {}
        void connect(PoolInterface *);
        void step();
        PoolSignals pull_pool();
    private:
        PoolSignals              ps = {0};
        PoolInterface           *connection = nullptr;
        ArbPrecData              pool_pixel;
        addr_t                   src_partition_size = 0;
        addr_t                   dst_partition_size = 0;
        int                      pool_cnt = 0;
        MemoryMap               *memory = nullptr;
};

class PoolArray {
    public:
        PoolArray(MemoryMap *mmap, size_t n_pools);
        Pool& operator[](int index);
        void connect(PoolInterface *);
        void step();
    private:
        std::vector<Pool>     pooler;

};
#endif
