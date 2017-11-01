#ifndef IO_H
#define IO_H


#include "tpb_isa.h"
#include <string>
#include <vector>

typedef union memory_accessor{
    char      *char_ptr;
    uint32_t  *uint32_ptr;
    int32_t   *int32_ptr;
    uint64_t  *uint64_ptr;
    int64_t   *int64_ptr;
    float     *fp32_ptr;
} __attribute((__packed__)) memory_accessor;

class Memory {
    public:
        Memory(size_t size);
        ~Memory();
        void *io_mmap(std::string fname, int &i, int &j, int &k, int &l, size_t &word_size);
        void io_write(std::string fname, void *ptr, int i, int j, int k, int l, ARBPRECTYPE dtype);
        //void munmap(addr_t addr);
        void read(void *dest, addr_t src, size_t n_bytes);
        void write(addr_t dest, void *src, size_t n_bytes);
        addr_t index(addr_t base, unsigned int i, ARBPRECTYPE dtype);
        void *translate(addr_t addr);
        void swap_axes(void *ptr, int i, int j, int k, int l, size_t word_size);
        void bank_mmap(addr_t addr, void *ptr, int count, size_t size);
        void *psum_bank_munmap(addr_t addr, int count, size_t size);
        void *sbuffer_bank_munmap(addr_t addr, int count, size_t size);
    private:
        char *memory;
        void *bank_munmap(addr_t addr, int count, addr_t stride, size_t size);

};

class MemoryMap; 

class MemoryMapInstance {
    public:
        MemoryMapInstance(MemoryMap *_mmap, addr_t _base, size_t _sz);
        void read_local(void *dest, addr_t src, size_t n_bytes);
        void write_local(addr_t dest, void *src, size_t n_bytes);
        void read_local_offset(void *dest, addr_t src_offset, size_t n_bytes);
        void write_local_offset(addr_t dest_offset, void *src, size_t n_bytes);
        addr_t get_base();
        size_t get_size();
    private:
        MemoryMap *mmap;
        addr_t  base;
        size_t  sz;
        bool in_range(addr_t addr, size_t n_bytes);
};

class MemoryMap {
    public:
        MemoryMap(Memory *_memory);
        MemoryMapInstance *mmap(addr_t addr, size_t sz);
        void read_global(void *dest, addr_t src, size_t n_bytes);
        void write_global(addr_t dest, void *src, size_t n_bytes);
    private:
        std::vector<MemoryMapInstance *> mmaps;
        Memory *memory;

};


#endif  //IO_H
