#ifndef _KERNELTESTER_H
#define _KERNELTESTER_H

#include "sigint.h"
#include "ibuffer.h"
#include <vector>

class KernelTester : public UopFeedInterface {
    public:
        KernelTester();
        ~KernelTester();
        bool         empty() {return ibuf.empty();}
        void        *front() {return ibuf.front();}
        void         pop()   {ibuf.pop();}
        void         convolve(char *i_name, char *f_name, char *o_name, 
                uint8_t padding[2]);

    private:
        IBuffer  ibuf;
        void    *ibuf_start;
        void    *ibuf_end;
        void get_dims(char *fname, uint64_t *shape, size_t *word_size);
};

#endif
