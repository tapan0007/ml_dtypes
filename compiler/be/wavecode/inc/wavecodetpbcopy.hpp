#pragma once

#ifndef KCC_WAVECODE_WAVECODETPBCOPY_H
#define KCC_WAVECODE_WAVECODETPBCOPY_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "compisa/inc/compisadmatrigger.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class TpbCopyWaveOp;
}

namespace wavecode {



class WaveCodeTpbCopy : public WaveCodeWaveOp {
public:
    using TransferRange = std::tuple<FileRange, TpbAddress>;
public:
    //----------------------------------------------------------------
    WaveCodeTpbCopy(WaveCodeRef waveCode);
    void rWaveCodeSbAtomLoadKelf(const WaveCodeSbAtomLoadKelf* waveCodeSbAtomLoadKelf)
    {
        m_WaveCodeSbAtomLoadKelf = waveCodeSbAtomLoadKelf;
    }

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

    const auto& gNotCopied() const {
        return m_NotCopiedFileToSbufAddress;
    }
    void writeDmaTriggerInstruction() const;

private:
    const WaveCodeSbAtomLoadKelf*   m_WaveCodeSbAtomLoadKelf = nullptr;
    std::set<TransferRange>         m_NotCopiedFileToSbufAddress;

    // To issue DMA TRIGGER after DMA_TRIGGER for parallel Load
    compisa::DmaTriggerInstr        m_DmaTriggerInstr;
    EngineId                        m_ChosenEngId;
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

