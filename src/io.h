#ifndef IO_H
#define IO_H


#include "types.h"
#include <string>

class Memory {
    public:
        Memory(size_t size);
        ~Memory();
        void *io_mmap(std::string fname, int &i, int &j, int &k, int &l, size_t &word_size);
        void io_write(std::string fname, void *ptr, int i, int j, int k, int l, size_t word_size);
        //void munmap(addr_t addr);
        void read(void *dest, addr_t src, size_t n_bytes);
        void write(addr_t dest, void *src, size_t n_bytes);
        void *translate(addr_t addr);
        void swap_axes(void *ptr, int i, int j, int k, int l, size_t word_size);
        void bank_mmap(addr_t addr, void *ptr, int count, size_t size);
        void *bank_munmap(addr_t addr, int count, size_t size);
    private:
        char *memory;

};


#endif  //IO_H
