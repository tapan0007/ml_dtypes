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
    bool generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp);
    void generateMatMul(wave::MatMulWaveOp* matmulWaveOp, bool noSyncNeededOnMatMulInstr);
    bool qLoadWeightsWaitsFor(const wave::WaveEdge* prevEdge) const;

    bool qSyncOnLdWeightsInstr(const wave::WaveEdge* prevEdge) const;
    bool qSyncOnMatMulInstr(const wave::WaveEdge* prevEdge) const;

    void countSyncedWeithsIfmapPred(wave::MatMulWaveOp* matmulWaveop,
                kcc_int32& numSyncedPrevWeights, kcc_int32& numSyncedPrevIfmaps);
private:
    bool m_SyncIfmapOnLdWeigthsInstr = false;
};


}}

#endif // KCC_WAVECODE_WAVECODEMATMUL_H
