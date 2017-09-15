#ifndef _IBUFFER_H
#define _IBUFFER_H

#include "sigint.h"

class IBuffer : public UopFeedInterface {
    public:
        bool         empty();
        void        *front();
        void         pop();
        void         setup(void *start, void *end);
    private:
        char   *cmem;
        char   *cmem_end;
};

#endif
