#include "pool.h"
#include "io.h"

extern Memory memory;
extern addr_t psum_buffer_base;

//------------------------------------------------------------
// Pool
//------------------------------------------------------------
Pool::Pool() : connection(NULL), base_ptr(mem) {
    ps = {0};
}

Pool::~Pool() {}

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
    ps = connection->pull_pool();
    if (!ps.valid) {
        return;
    }
	addr_t src_partition_size;
	addr_t dst_partition_size;
	size_t dsize = sizeofArbPrecType(ps.dtype);

	ps.countdown--;
	
	src_partition_size = (ps.src_full_addr >= psum_buffer_base) ?
		Constants::psum_addr : Constants::partition_nbytes;
	dst_partition_size = (ps.dst_full_addr >= psum_buffer_base) ?
		Constants::psum_addr : Constants::partition_nbytes;

    if (ps.start) {
        curr_ptr = base_ptr;
    }
    memory.read(curr_ptr, ps.src_full_addr, dsize);
    curr_ptr += dsize;

    if (ps.stop) {
        assert(curr_ptr > base_ptr && "empty pool?");
        switch (ps.func) {
            case IDENTITY_POOL:
                memory.write(ps.dst_full_addr, base_ptr, curr_ptr - base_ptr);
                break;

#if 0
            case AVG_POOL:
                // fixme - how can we divide with just a multiplying unit?
                //double pool_pixel = pool_pixel * ArbPrecType(ew.psum_dtype, (1.0 / (ew.pool_dimx * ew.pool_dimy)));
                for (int i = 0; i < ew.pool_dimx; i++) {
                    for (int j = 0; j < ew.pool_dimy; j++) {
                        pool_pixel = pool_pixel + entry[e_id - i * ew.pool_stride - j].partial_sum;
                    }
                }
                pool_pixel = pool_pixel / ArbPrec(ew.psum_dtype, n);
                break;
            case MAX_POOL:
                pool_pixel = entry[e_id].partial_sum;
                for (int i = 0; i < ew.pool_dimx; i++) {
                    for (int j = 0; j < ew.pool_dimy; j++) {
                        ArbPrecData comp_pixel = entry[e_id - i * Constants::partition_nbytes - j].partial_sum;
                        if (comp_pixel > pool_pixel) {
                            pool_pixel = comp_pixel;
                        }
                    }
                }
                break;
#endif
            default:
                assert(0 && "that pooling is not yet supported");
                break;
        }
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
