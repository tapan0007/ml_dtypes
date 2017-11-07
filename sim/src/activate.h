#ifndef ACTIVATE_H
#define ACTIVATE_H

#include "sigint.h"
#include <vector>

class MemoryMap;

class Activate : public ActivateInterface{
    public:
        Activate(MemoryMap *_memory) : memory(_memory) {};
        ActivateSignals pull_activate();
        void connect(ActivateInterface *);
        void step();
    private:
        MemoryMap           *memory;
        ActivateInterface   *connection;
        ActivateSignals      as = {0};

};

class ActivateArray {
    public:
        ActivateArray(MemoryMap *memory, size_t n_cols = 64);
        Activate& operator[](int index);
        void connect(ActivateInterface *);
        void step();
    private:
        std::vector<Activate>     activators;

};
#endif
