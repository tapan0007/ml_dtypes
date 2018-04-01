#pragma once

#ifndef KCC_COMPISA_WRITE_H
#define KCC_COMPISA_WRITE_H



#include "shared/inc/tpb_isa_write.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class WriteInstr : public WRITE {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    WriteInstr()
        : WRITE()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_WRITE_H

