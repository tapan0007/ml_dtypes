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
    ArbPrecData raw_pixel;
    ArbPrecData in_pixel;
    in_pixel.raw = 0;
    ps = connection->pull_pool();
    if (!ps.valid) {
        return;
    }
    ARBPRECTYPE in_dtype  = ps.in_dtype;
    ARBPRECTYPE out_dtype = ps.out_dtype;
    size_t in_dsize  = sizeofArbPrecType(ps.in_dtype);
    size_t out_dsize = sizeofArbPrecType(ps.out_dtype);


    memory->read_global(&raw_pixel, ps.src_addr.sys, in_dsize);
    if (ps.func == AVG_POOL) {
        in_pixel = ArbPrec::cast_to_fp32(raw_pixel, in_dtype);
    }  else {
        in_pixel = raw_pixel;
    }

    /* start ! */
    if (ps.start) {
        pool_pixel.raw = 0;
        averager_pixel.fp32 = 1.0/ps.avg_count;
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
            {
                ARBPRECTYPE dtype_fp32 = ARBPRECTYPE::FP32;
                ArbPrecData scale_pixel = ArbPrec::multiply(in_pixel, 
                        averager_pixel, dtype_fp32, dtype_fp32);
                pool_pixel = ArbPrec::add(pool_pixel, scale_pixel, dtype_fp32);
            }
            break;
        case MAX_POOL:
            if (ps.start) {
                pool_pixel = in_pixel;
            } else {
                if (ArbPrec::gt(in_pixel, pool_pixel, in_dtype)) {
                    pool_pixel = in_pixel;
                }
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
                pool_pixel = ArbPrec::cast_from_fp32(pool_pixel, out_dtype);
                break;
            case MAX_POOL:
                break;
            default:
                assert(0 && "that pooling is not yet supported");
                break;
        }
        memory->write_global(ps.dst_addr.sys, &pool_pixel, out_dsize);
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
