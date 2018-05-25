#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMLOAD_H
#define KCC_WAVECODE_WAVECODESBATOMLOAD_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodesbatom.hpp"


namespace kcc {

namespace wave {
    class SbAtomLoadWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomLoad : public WaveCodeSbAtom {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomLoad(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
    void generateForSim(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    void generateDmaCopySimKelf(wave::SbAtomLoadWaveOp*sbAtomLoadWaveop,
            EngineId chosenEngId,
            const std::vector<events::EventId>& succEventIds);

    void generateDmaDescAndTriggerRuntimeKelf(wave::SbAtomLoadWaveOp*sbAtomLoadWaveop,
            EngineId chosenEngId,
            const std::vector<events::EventId>& succEventIds);

    kcc_int32 generateForKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    void generateDmaTrigger(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp,
                EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);

    void generateInputDma(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    kcc_int32 findSuccEventsAndChosenEngine(wave::SbAtomWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId, std::vector<events::EventId>& succEventIds);
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

