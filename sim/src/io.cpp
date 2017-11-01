#include "io.h"
#include "cnpy.h"
#include "uarch_cfg.h"
#include <complex>
#include <string.h>


Memory::Memory(size_t size) {
    memory = (char *)calloc(1, size);
}

Memory::~Memory() {
    free(memory);
}

/* go from mcrs->crsm */
void
Memory::swap_axes(void *ptr, int m, int c, int r, int s, size_t word_size)
{
    int nbytes = m * c * r * s * word_size;
    char *dest = (char *)ptr;
    char *src = (char *)malloc(nbytes);
    int   i = 0;
    memcpy(src, ptr, nbytes);

    for (int cc = 0; cc < c; cc++) {
        for (int rr = 0; rr < r; rr++) {
            for (int ss = 0; ss < s; ss++) {
                for (int mm = 0; mm < m; mm++) {
                    memcpy(&dest[i], &src[(mm*c*r*s + cc*r*s + rr*s + ss) * 
                            word_size], word_size);
                    i += word_size;
                }
            }
        }
    }
    free(src);
}

void * 
Memory::io_mmap(std::string fname, int &i, int &j, int &k, int &l, size_t &word_size) {
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    std::vector<unsigned int> &sh = arr.shape;
    i = sh[0];
    j = sh[1];
    k = sh[2];
    l = sh[3];
    word_size = arr.word_size;

    return arr.data;
}

void 
Memory::bank_mmap(addr_t addr, void *ptr, int count, size_t n_bytes)
{
    char *cptr = (char *)(ptr);
    assert((n_bytes  <= SZ(ROW_SIZE_BITS)) && "won't fit in partition");
    for (int i = 0; i < count; i++) {
        memcpy(memory + addr, cptr, n_bytes);
        addr  += SZ(ROW_SIZE_BITS);
        cptr  += n_bytes;
    }
}

void  *
Memory::bank_munmap(addr_t addr, int count, addr_t stride, size_t n_bytes)
{
    int tot_n_bytes = count * n_bytes;
    void *ptr = malloc(tot_n_bytes);
    char *cptr = (char *)ptr;
    for (int i = 0; i < count; i++) {
        memcpy(cptr, memory + addr, n_bytes);
        addr  += stride;
        cptr  += n_bytes;
    }
    return ptr;
}

void  *
Memory::sbuffer_bank_munmap(addr_t addr, int count, size_t n_bytes)
{
    return bank_munmap(addr, count, SZ(ROW_SIZE_BITS), n_bytes);
}

void  *
Memory::psum_bank_munmap(addr_t addr, int count, size_t n_bytes)
{
    return bank_munmap(addr, count, SZ(COLUMN_SIZE_BITS), n_bytes);
}

void 
Memory::read(void *dest, addr_t src, size_t n_bytes)
{
    memcpy(dest, memory + src, n_bytes);
}

void 
Memory::write(addr_t dest, void *src, size_t n_bytes)
{
    memcpy(memory + dest, src, n_bytes);
}

addr_t
Memory::index(addr_t base, unsigned int i, ARBPRECTYPE dtype)
{
    memory_accessor ma;
    ma.char_ptr = &memory[base];
    void *ptr;

    switch (dtype) {
        case UINT8:
            ptr = &ma.char_ptr[i];
            break;
        case UINT32:
            ptr = &ma.uint32_ptr[i];
            break;
        case INT32:
            ptr = &ma.int32_ptr[i];
            break;
        case UINT64:
            ptr = &ma.uint64_ptr[i];
            break;
        case INT64:
            ptr = &ma.int64_ptr[i];
            break;
        case FP32:
            ptr = &ma.fp32_ptr[i];
            break;
        default:
            assert(0);
    }
    return ((char *)ptr - memory);
}

void *
Memory::translate(addr_t addr) {
    return memory + addr;

}

void
Memory::io_write(std::string fname, void *ptr, int i,int j,int k,int l, ARBPRECTYPE dtype) {
    const unsigned int shape[] = {(unsigned int)i, (unsigned int)j, (unsigned int)k, (unsigned int)l};
    int word_size = sizeofArbPrecType(dtype);
    switch (dtype) {
        case UINT8:
            cnpy::npy_save(fname, (uint8_t *)ptr, shape, 4, word_size, "w");
            break;
        case UINT16:
            cnpy::npy_save(fname, (uint16_t *)ptr, shape, 4, word_size, "w");
            break;
        case UINT32:
            cnpy::npy_save(fname, (uint32_t *)ptr, shape, 4, word_size, "w");
            break;
        case FP16:
            cnpy::npy_save(fname, (float *)ptr, shape, 4, 2, "w");
            break;
        default:
            assert(0);
    }
}

/*---------------------------------------------*/
/* MemoryMap */
/*---------------------------------------------*/

MemoryMap::MemoryMap(Memory *_memory) : memory(_memory) {}

MemoryMapInstance *
MemoryMap::mmap(addr_t addr, size_t sz) {
    /* TODO -  check to make sure range is not already mmapd */
    MemoryMapInstance *mmap = new MemoryMapInstance(this, addr, sz);
    mmaps.push_back(mmap);
    return mmap;
}

void
MemoryMap::read_global(void *dest, addr_t src, size_t n_bytes) {
    /* FIXME - check that we aren't in reserved range */
    return memory->read(dest, src, n_bytes);
}


void 
MemoryMap::write_global(addr_t dest, void *src, size_t n_bytes) {
    /* FIXME - check that we aren't in reserved range */
    return memory->write(dest, src, n_bytes);
}

/*---------------------------------------------*/
/* MemoryMapInstance */
/*---------------------------------------------*/

MemoryMapInstance::MemoryMapInstance(MemoryMap *_mmap, addr_t _base, size_t _sz) 
    : mmap(_mmap), base(_base), sz(_sz) {
}

addr_t
MemoryMapInstance::get_base() {
    return base;
}

size_t
MemoryMapInstance::get_size() {
    return sz;
}

bool
MemoryMapInstance::in_range(addr_t addr, size_t nbytes) {
    return ((addr >= base) && (addr + nbytes <= base + sz));
}


void
MemoryMapInstance::read_local(void *dest, addr_t src, size_t n_bytes) {
    assert(in_range(src, n_bytes));
    return mmap->read_global(dest, src, n_bytes);
}


void 
MemoryMapInstance::write_local(addr_t dest, void *src, size_t n_bytes) {
    assert(in_range(dest, n_bytes));
    return mmap->write_global(dest, src, n_bytes);
}

void
MemoryMapInstance::read_local_offset(void *dest, addr_t src_offset, size_t n_bytes) {
    return read_local(dest, base +src_offset, n_bytes);
}


void 
MemoryMapInstance::write_local_offset(addr_t dest_offset, void *src, size_t n_bytes) {
    return write_local(base + dest_offset, src, n_bytes);
}

