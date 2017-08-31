#ifndef POOL_BUFFER_H
#define POOL_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

class Pool : public PoolInterface {
    public:
        Pool();
        ~Pool();
        PoolSignals pull_pool();
        void connect(PoolInterface *);
        void step();
    private:
        PoolSignals              ps;
        PoolInterface            *connection;

};

class PoolArray {
    public:
        PoolArray(int n_pools = 64);
        ~PoolArray();
        Pool& operator[](int index);
        void connect(PoolInterface *);
        void step();
    private:
        std::vector<Pool>     pooler;

};
#endif 
