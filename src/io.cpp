#include "io.h"
#include "types.h"
#include "cnpy.h"
#include <complex>


void *
io_mmap(std::string fname, int &i, int &j, int &k, int &l) {
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    std::vector<unsigned int> &sh = arr.shape;
    i = sh[0];
    j = sh[1];
    k = sh[2];
    l = sh[3];

    return static_cast<void *>(arr.data);
}

void munmap(void *addr) {
    delete [] (char *)addr;
}


