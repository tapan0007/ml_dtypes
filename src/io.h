#ifndef IO_H
#define IO_H


#include "types.h"
#include <string>

class Memory {
    public:
        Memory(int size);
        ~Memory();
        int io_mmap(addr_t dest, std::string fname, int &i, int &j, int &k, int &l);
        void io_write(std::string fname, addr_t addr, int i, int j, int k, int l, ARBPRECTYPE type);
        //void munmap(addr_t addr);
        void read(void *dest, addr_t src, int n_bytes);
        void write(addr_t dest, void *src, int n_bytes);
        void *translate(addr_t addr);
        void swap_axes(addr_t addr, int i, int j, int k, int l, int n_bytes);
    private:
        char *memory;

};


#endif  //IO_H
