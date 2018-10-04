#pragma once

#ifndef KCC_WAVECODE_WAVECODETENSORTENSOR_H
#define KCC_WAVECODE_WAVECODETENSORTENSOR_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class TensorTensorWaveOp;
}

namespace wavecode {



class WaveCodeTensorTensor : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeTensorTensor(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

private:
    void generateDiffBufSrc(wave::TensorTensorWaveOp* tensortensorWaveop);
    void generateSameBufSrc(wave::TensorTensorWaveOp* tensortensorWaveop);
};


}}

#endif // KCC_WAVECODE_WAVECODETENSORTENSOR_H

