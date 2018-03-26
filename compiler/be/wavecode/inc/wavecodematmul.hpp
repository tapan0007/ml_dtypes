#pragma once

#ifndef KCC_WAVECODE_WAVECODEMATMUL_H
#define KCC_WAVECODE_WAVECODEMATMUL_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class MatMulWaveOp;
}

namespace wavecode {



class WaveCodeMatMul : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeMatMul(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

private:
    void generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp);
    void generateMatMul(wave::MatMulWaveOp* matmulWaveOp);
};


}}

#endif // KCC_WAVECODE_WAVECODEMATMUL_H
