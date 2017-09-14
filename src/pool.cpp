#include "pool.h"
#include "io.h"

extern Memory memory;
extern addr_t psum_buffer_base;

//------------------------------------------------------------
// Pool
//------------------------------------------------------------
void
Pool::connect(PoolInterface *_connection) {
    connection = _connection;
}

PoolSignals
Pool::pull_pool() {
    return ps;
}

void
Pool::step() {
    ArbPrecData in_pixel = {0};
    ps = connection->pull_pool();
    if (!ps.valid) {
        return;
    }
	addr_t src_partition_size;
	addr_t dst_partition_size;
	ARBPRECTYPE dtype = ps.dtype;
	size_t dsize = sizeofArbPrecType(ps.dtype);

	ps.countdown--;
	
	src_partition_size = (ps.src_full_addr >= psum_buffer_base) ?
		Constants::psum_addr : Constants::partition_nbytes;
	dst_partition_size = (ps.dst_full_addr >= psum_buffer_base) ?
		Constants::psum_addr : Constants::partition_nbytes;

    memory.read(&in_pixel, ps.src_full_addr, dsize);

    /* start ! */
    if (ps.start) {
        pool_pixel = {0};
        pool_cnt = 0;
    }
    switch (ps.func) {
        case IDENTITY_POOL:
            break;
        case AVG_POOL:
            pool_pixel = ArbPrec::add(&pool_pixel, &in_pixel, dtype);
            pool_cnt++;
            break;
        case MAX_POOL:
            if (ArbPrec::gt(&pool_pixel, &in_pixel, dtype)) {
                pool_pixel = in_pixel;
            }
            break;
        default:
            assert(0 && "that pooling is not yet supported");
            break;
    }

    if (ps.stop) {
        switch (ps.func) {
            case IDENTITY_POOL:
                assert(ps.start == ps.stop);
                break;
            case AVG_POOL:
                pool_pixel = ArbPrec::uint_divide(&pool_pixel, pool_cnt, dtype);
                break;
            case MAX_POOL:
                break;
            default:
                assert(0 && "that pooling is not yet supported");
                break;
        }
        memory.write(ps.dst_full_addr, &in_pixel, dsize);
    }
    ps.src_full_addr += src_partition_size;
    ps.dst_full_addr += dst_partition_size;
    if (!ps.countdown) {
        ps.valid = false;
    }
}

#if 0
    ArbPrecData
        Pool::activation(ArbPrecData pixel) {
            switch (ew.activation) {
                case RELU:
                    break;
                case LEAKY_RELU:
                    break;
                case SIGMIOD:
           break;
        case TANH:
           break;
        case IDENTITY:
           break;
        default:
           break;
    }
    return pixel;
}
#endif



//------------------------------------------------------------
// PoolBufferArray
//------------------------------------------------------------
PoolArray::PoolArray(int n_pools) {
    pooler.resize(n_pools);
    for (int i = 1; i < n_pools; i++) {
        pooler[i].connect(&pooler[i-1]);
    }
}

PoolArray::~PoolArray() {}

void
PoolArray::connect(PoolInterface *connection) {
    pooler[0].connect(connection);
}


Pool& 
PoolArray::operator[](int index)
{
    return pooler[index];
}

void
PoolArray::step() {
    int n_pools = pooler.size();
    for (int i = n_pools - 1; i >= 0; i--) {
        pooler[i].step();
    }
}
