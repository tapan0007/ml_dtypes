#pragma once

#ifndef KCC_WAVECODE_WAVECODEMATMUL_H
#define KCC_WAVECODE_WAVECODEMATMUL_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace wave {
    class MatMulWaveOp;
}

namespace wavecode {



class WaveCodeMatMul : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeMatMul(WaveCode* wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

private:
    void generateLoadWeights(MatMullWaveOp *matmulWaveOp, bool firstWeight);
};


}}

#endif // KCC_WAVECODE_WAVECODEMATMUL_H
