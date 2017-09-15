#include "ibuffer.h"
#include "sigint.h"
#include "types.h"

void
IBuffer::setup(void *start, void *end) {
    cmem = (char *)start;
    cmem_end = (char *)end;
}

bool 
IBuffer::empty() {
    return cmem == cmem_end;
}

void * 
IBuffer::front() {
    return (void *)(cmem);
}

void
IBuffer::pop() {
    cmem += INSTRUCTION_NBYTES;
}

