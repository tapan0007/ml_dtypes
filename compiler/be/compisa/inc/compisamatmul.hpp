#pragma once

#ifndef KCC_COMPISA_MATMUL_H
#define KCC_COMPISA_MATMUL_H



#include "shared/inc/tpb_isa_matmul.hpp"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {


class MatMulInstr : public MATMUL {
public:
    static constexpr EngineId engineId = EngineId::PeArray;
public:
    //----------------------------------------------------------------
    MatMulInstr()
        : MATMUL()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_MATMUL_H


