#include "io.h"
#include "types.h"
#include "cnpy.h"
#include <complex>
#include <string.h>


Memory::Memory(size_t size) {
    memory = (char *)calloc(1, size);
}

Memory::~Memory() {
    free(memory);
}

void
Memory::swap_axes(void *ptr, int r, int s, int t, int u, size_t word_size)
{
    int nbytes = r * s * t * u * word_size;
    char *dest = (char *)ptr;
    char *src = (char *)malloc(nbytes);
    int   a = 0;
    int   stride = nbytes / (r * s * t * u);
    int   step = t * u * stride;
    memcpy(src, ptr, nbytes);

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < r; j++) {
            memcpy(&dest[a], &src[(i + j * s) * step], step);
            a += step;
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
    assert((n_bytes * count <= Constants::bytes_per_bank) && "won't fit in bank");
    for (int i = 0; i < count; i++) {
        memcpy(memory + addr, cptr, n_bytes);
        addr  += Constants::partition_nbytes;
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
    return bank_munmap(addr, count, Constants::partition_nbytes, n_bytes);
}

void  *
Memory::psum_bank_munmap(addr_t addr, int count, size_t n_bytes)
{
    return bank_munmap(addr, count, Constants::psum_addr, n_bytes);
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
Memory::io_write(std::string fname, void *ptr, int i,int j,int k,int l, size_t word_size) {
    const unsigned int shape[] = {(unsigned int)i, (unsigned int)j, (unsigned int)k, (unsigned int)l};
    switch (word_size) {
        case 1:
            cnpy::npy_save(fname, (uint8_t *)ptr, shape, 1, "w");
            break;
        case 4: // good for FP too?
            cnpy::npy_save(fname, (uint32_t *)ptr, shape, 4, "w");
            break;
        case 8: // good for FP too?
            cnpy::npy_save(fname, (uint64_t *)ptr, shape, 8, "w");
            break;
        default:
            assert(0);
    }
}
