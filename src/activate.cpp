#include "activate.h"

//-----------------------------------------------------------------------
//  Activate
//-----------------------------------------------------------------------
Activate::Activate() : north(nullptr) {}

Activate::~Activate() {}

void
Activate::connect_north(PSumActivateInterface *_north)
{
    north = _north;
}

void
Activate::step()
{
    ps = north->pull_psum();
    if (ps.valid) {
        printf("Activate ");
        ps.partial_sum.dump(stdout);
        printf("\n");
    }
}

//-----------------------------------------------------------------------
//  ActivateArray
//-----------------------------------------------------------------------
ActivateArray::ActivateArray(int n_cols) {
    buffer.resize(n_cols);
}

ActivateArray::~ActivateArray() {}

void
ActivateArray::connect_north(int id, PSumActivateInterface *_north)
{
    buffer[id].connect_north(_north);
}

void
ActivateArray::step() {
    for (int i = buffer.size() - 1; i >= 0; i--) {
        buffer[i].step();
    }
}
