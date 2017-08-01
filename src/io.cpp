#include "io.h"
#include "types.h"
#include "cnpy.h"
#include <complex>
#include <string.h>

Memory::Memory(int size) {
    memory = (char *)calloc(1, size);
}

Memory::~Memory() {
    free(memory);
}

int 
Memory::io_mmap(addr_t dest, std::string fname, int &i, int &j, int &k, int &l) {
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    std::vector<unsigned int> &sh = arr.shape;
    int n_bytes;
    i = sh[0];
    j = sh[1];
    k = sh[2];
    l = sh[3];
    n_bytes = i * j * k * l * arr.word_size;

    memcpy(memory + dest, arr.data, n_bytes);
    return n_bytes;
}

void 
Memory::read(void *dest, addr_t src, int n_bytes)
{
    memcpy(dest, memory + src, n_bytes);
}

void 
Memory::write(addr_t dest, void *src, int n_bytes)
{
    memcpy(memory + dest, src, n_bytes);
}

void *
Memory::translate(addr_t addr) {
    return memory + addr;

}



