#include "activate.h"

//-----------------------------------------------------------------------
//  Activate
//-----------------------------------------------------------------------
Activate::Activate() : psum_connect(nullptr) {}

Activate::~Activate() {}

void
Activate::connect_psum(PSumActivateInterface *_psum_connect)
{
    psum_connect = _psum_connect;
}

void
Activate::step()
{
    ps = psum_connect->pull_psum();
    if (ps.valid) {
        printf("Activate ");
        ArbPrec::dump(stdout, ps.partial_sum, ps.dtype);
        printf("\n");
    }
}

ActivateSbSignals 
Activate::pull_activate()
{
    return ActivateSbSignals{ps.valid, ps.partial_sum};
}

//-----------------------------------------------------------------------
//  ActivateArray
//-----------------------------------------------------------------------
ActivateArray::ActivateArray(int n_cols) {
    buffer.resize(n_cols);
}

ActivateArray::~ActivateArray() {}

Activate& ActivateArray::operator[](int index){
    return buffer[index];
}

void
ActivateArray::connect_psum(int id, PSumActivateInterface *_psum)
{
    buffer[id].connect_psum(_psum);
}

void
ActivateArray::step() {
    for (int i = buffer.size() - 1; i >= 0; i--) {
        buffer[i].step();
    }
}
