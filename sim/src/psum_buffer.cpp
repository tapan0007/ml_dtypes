#include "psum_buffer.h"
#include "string.h"
#include "io.h"

//------------------------------------------------------------
// PsumBuffer
//------------------------------------------------------------
PSumBuffer::PSumBuffer(MemoryMap *mmap, addr_t base, size_t size) {
    mem = mmap->mmap(base, size);
}

void
PSumBuffer::connect_west(EdgeInterface *_west) {
    west = _west;
}

void
PSumBuffer::connect_north(PeNSInterface *_north) {
    north = _north;
}

EdgeSignals
PSumBuffer::pull_edge() {
    return ew;
}

/* FIXME - dummy function for now */
PSumActivateSignals
PSumBuffer::pull_psum() {
    return PSumActivateSignals{false, ArbPrecData(), INVALID_ARBPRECTYPE};
}


void
PSumBuffer::step() {
    ns = north->pull_ns();
    ew = west->pull_edge();
    static ArbPrecData zeros;
    static ArbPrecData ones;
    ArbPrecData src;
    void *src_ptr = &src;
    uint8_t valid;
    ones.raw = 0xffffffffffffffff;

    if (ew.column_valid && (ew.ifmap_valid || ew.pad_valid)) {
        assert(ew.psum_addr.sys >= MMAP_PSUM_BASE);
        ARBPRECTYPE psum_dtype =  get_upcast(ew.ifmap_dtype);
        size_t dsize = sizeofArbPrecType(psum_dtype);

        addr_t src_off = ew.psum_addr.column;
        addr_t meta_off = src_off | (1 << (COLUMN_BYTE_OFFSET_BITS +
                    BANKS_PER_COLUMN_BITS));
        mem->read_local_offset(src_ptr, src_off, dsize);

        if (ew.psum_start) {
            mem->write_local_offset(src_off, &zeros, dsize);
            mem->write_local_offset(meta_off, &ones, dsize);
        }
        if (ew.ifmap_valid) {
            printf("adding partial sum at %x is ", mem->get_base() | src_off);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            ArbPrecData result = ArbPrec::add(src, ns.partial_sum, psum_dtype);
            mem->write_local_offset(src_off, &result, dsize);
            printf(" total = ");
            ArbPrec::dump(stdout, result, psum_dtype);
            printf("\n");
            mem->read_local_offset(&valid, meta_off, 1);
            assert(valid);
        }

        if (ew.psum_stop) {
            mem->read_local_offset(src_ptr, src_off, dsize);
            printf("final partial sum at %x is ", mem->get_base() | src_off);
            ArbPrec::dump(stdout, src, psum_dtype);
            printf("\n");
            mem->read_local_offset(&valid, meta_off, 1);
            assert(valid);
            mem->write_local_offset(meta_off, &zeros, 1);
        }

        if (ew.column_valid) {
            ew.column_valid = ((ew.column_countdown--) > 0);
        }
    }
}

//------------------------------------------------------------
// PsumBufferArray
//------------------------------------------------------------
PSumBufferArray::PSumBufferArray(MemoryMap *mmap, addr_t base, size_t n_cols) {
    size_t col_sz = SZ(COLUMN_SIZE_BITS);
    for (size_t i = 0; i < n_cols; i++) {
        col_buffer.push_back(PSumBuffer(mmap, base, col_sz));
        base += col_sz;
    }
    for (size_t i = 1; i < n_cols; i++) {
        col_buffer[i].connect_west(&col_buffer[i-1]);
    }
}

PSumBufferArray::~PSumBufferArray() {}

void
PSumBufferArray::connect_west(EdgeInterface *west) {
    col_buffer[0].connect_west(west);
}

void
PSumBufferArray::connect_north(int col, PeNSInterface *north) {
    col_buffer[col].connect_north(north);
}

PSumBuffer&
PSumBufferArray::operator[](int index)
{
    return col_buffer[index];
}

void
PSumBufferArray::step() {
    int n_cols = col_buffer.size();
    for (int i = n_cols - 1; i >= 0; i--) {
        col_buffer[i].step();
    }
}
