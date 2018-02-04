#pragma once

#ifndef KCC_WAVE_MATMULWAVEOP_H
#define KCC_WAVE_MATMULWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/waveop.hpp"


namespace kcc {
namespace wave {


class MatMulWaveOp : public WaveOp {
public:
    MatMulWaveOp(const Params& params, const FmapDesc& fmapDesc,
        const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qMatMultWaveOp() const override {
        return true;
    }

};

}}


#endif


