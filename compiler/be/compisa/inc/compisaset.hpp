#pragma once

#ifndef KCC_COMPISA_SET_H
#define KCC_COMPISA_SET_H



#include "shared/inc/tpb_isa_set.hpp"


namespace kcc {

namespace compisa {


class SetInstr : public SET {
public:
    //----------------------------------------------------------------
    SetInstr()
        : SET()
    {
    }

};


}}

#endif // KCC_COMPISA_SET_H

