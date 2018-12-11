#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMLOAD_KELF_H
#define KCC_WAVECODE_WAVECODESBATOMLOAD_KELF_H

#include <string>
#include <cstdio>
#include <map>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

//#include "compisa/inc/compisasimmemcpy.hpp"

#include "wavecode/inc/wavecodesbatomload.hpp"


namespace kcc {

namespace wave {
    class SbAtomLoadWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomLoadKelf : public WaveCodeSbAtomLoad {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomLoadKelf(WaveCodeRef waveCode);

    void rWaveCodeTpbCopy(const WaveCodeTpbCopy* waveCodeTpbCopy) {
        m_WaveCodeTpbCopy = waveCodeTpbCopy;
    }

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

    bool Loaded(const FileRange& tongaRange, TpbAddress& srcAddress, OffsetRange& loadedRange) const;

private:
    void generateDmaDescAndTriggerRuntimeKelf(wave::SbAtomLoadWaveOp*sbAtomLoadWaveop,
            EngineId chosenEngId,
            const std::vector<events::EventId>& succEventIds);

    void generateDmaDescAndTriggerRuntimeKelfWithReplication(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);

    kcc_int32 generateForKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    void generateDmaTrigger(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp,
                EngineId chosenEngId, const std::vector<events::EventId>& succEventIds);

    void generateInputDma(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    void generateInputDmaRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);
    void generateInputDmaReplWithCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop);
    void generateInputDmaReplWithoutCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop);

    void generateInputDmaNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);


private:

    const WaveCodeTpbCopy*          m_WaveCodeTpbCopy = nullptr;
    std::map<FileRange, TpbAddress> m_LoadedFileToSbufAddress;
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

