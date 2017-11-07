#include "activate.h"

//-----------------------------------------------------------------------
//  Activate
//-----------------------------------------------------------------------
void
Activate::connect(ActivateInterface *connection)
{
    this->connection = connection;
}

ActivateSignals
Activate::pull_activate()
{
    return as;
}

void
Activate::step()
{
    as = connection->pull_activate();
    if (as.valid) {
        printf("Activate\n");
        //ArbPrec::dump(stdout, as.partial_sum, as.dtype);
        //printf("\n");
        as.valid = ((as.countdown--) > 0);
    }
}

//-----------------------------------------------------------------------
//  ActivateArray
//-----------------------------------------------------------------------
ActivateArray::ActivateArray(MemoryMap *mmap, size_t n_cols) {
    for (size_t i = 0; i < n_cols; i++) {
        activators.push_back(Activate(mmap));
    }
    for (size_t i = 1; i < activators.size(); i++) {
        activators[i].connect(&activators[i-1]);
    }
}

Activate& ActivateArray::operator[](int index){
    return activators[index];
}

void
ActivateArray::connect(ActivateInterface *ai)
{
    activators[0].connect(ai);
}

void
ActivateArray::step() {
    for (int i = activators.size() - 1; i >= 0; i--) {
        activators[i].step();
    }
}
