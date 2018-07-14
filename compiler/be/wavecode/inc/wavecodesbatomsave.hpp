#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMSAVE_H
#define KCC_WAVECODE_WAVECODESBATOMSAVE_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"

#include "wavecode/inc/wavecodesbatom.hpp"


namespace kcc {

namespace wave {
    class SbAtomSaveWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomSave : public WaveCodeSbAtom {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomSave(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;



private:
    void generateForSim(wave::SbAtomSaveWaveOp* waveOp);
    void generateForKelf(wave::SbAtomSaveWaveOp* waveOp);
    //void generateDmaTrigger(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
    //                EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);
    void generateDmaTriggerRuntimeKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);
    void generateDmaCopySimKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);
    kcc_int32 findSuccEventsAndChosenEngine(wave::SbAtomSaveWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId, std::vector<events::EventId>& succEventIds);

    void calcOutputSize(const wave::SbAtomSaveWaveOp* sbAtomSaveWaveop);
};


}}

#endif // KCC_WAVECODE_WAVECODESBATOMSAVE_H

