#include "pool.h"
#include "io.h"

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
    ArbPrecData in_pixel;
    in_pixel.raw = 0;
    ps = connection->pull_pool();
    if (!ps.valid) {
        return;
    }
    ARBPRECTYPE dtype = ps.dtype;
    ARBPRECTYPE up_dtype;
    size_t dsize = sizeofArbPrecType(ps.dtype);


    memory->read_global(&in_pixel, ps.src_addr.sys, dsize);

    /* start ! */
    if (ps.start) {
        pool_pixel.raw = 0;
        pool_cnt = 0;
        /* FIXME - how is this going to be done in HW?
         * Should we put check to make sure we stay in bounds? */
        src_partition_size = (ps.src_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
        dst_partition_size = (ps.dst_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
    }
    switch (ps.func) {
        case IDENTITY_POOL:
            pool_pixel = in_pixel;
            break;
        case AVG_POOL:
            up_dtype = get_upcast(ps.dtype);
            pool_pixel = ArbPrec::add(pool_pixel, in_pixel, up_dtype);
            pool_cnt++;
            break;
        case MAX_POOL:
            if (ArbPrec::gt(in_pixel, pool_pixel, dtype)) {
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
                pool_pixel = ArbPrec::uint_divide(pool_pixel, pool_cnt,
                        up_dtype);
                break;
            case MAX_POOL:
                break;
            default:
                assert(0 && "that pooling is not yet supported");
                break;
        }
        memory->write_global(ps.dst_addr.sys, &pool_pixel, dsize);
    }
    ps.src_addr.sys += src_partition_size;
    ps.dst_addr.sys += dst_partition_size;
    if (ps.valid) {
        ps.valid = ((ps.countdown--) > 0);
    }
}

//------------------------------------------------------------
// PoolBufferArray
//------------------------------------------------------------
PoolArray::PoolArray(MemoryMap *mmap, size_t n_pools) {
    for (size_t i = 0; i < n_pools; i++) {
        pooler.push_back(Pool(mmap));
    }
    for (size_t i = 1; i < n_pools; i++) {
        pooler[i].connect(&pooler[i-1]);
    }
}

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
