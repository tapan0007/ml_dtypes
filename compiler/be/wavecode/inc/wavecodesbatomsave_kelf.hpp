#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMSAVE_KELF_H
#define KCC_WAVECODE_WAVECODESBATOMSAVE_KELF_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"

#include "wavecode/inc/wavecodesbatomsave.hpp"


namespace kcc {

namespace wave {
    class SbAtomSaveWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomSaveKelf : public WaveCodeSbAtomSave {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomSaveKelf(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;



private:
    void generateForKelf(wave::SbAtomSaveWaveOp* waveOp);
    void generateDmaTriggerRuntimeKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);
};


}}

#endif // KCC_WAVECODE_WAVECODESBATOMSAVE_H

