#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMSAVE_SIM_H
#define KCC_WAVECODE_WAVECODESBATOMSAVE_SIM_H

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



class WaveCodeSbAtomSaveSim : public WaveCodeSbAtomSave {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomSaveSim(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;



private:
    void generateForSim(wave::SbAtomSaveWaveOp* waveOp);
};


}}

#endif // KCC_WAVECODE_WAVECODESBATOMSAVE_H

