#pragma once

#ifndef KCC_SERIALIZE_SERWAVEOP_H
#define KCC_SERIALIZE_SERWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>




#include "utils/inc/debug.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomwaveop.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


class SerWaveOp {
public:
    SerWaveOp();


    SerWaveOp(const SerWaveOp&) = default;

public:
    template<typename Archive>
    void serialize(Archive & archive)
    {
    }

private:
}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

