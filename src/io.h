#ifndef IO_H
#define IO_H

#include "cnpy.h"
#include "types.h"
void *io_mmap(std::string fname, int &i, int &j, int &k, int &l);
void munmap(void *addr);


#endif  //IO_H
