#pragma once

#ifndef KCC_WAVECODE_WAVECODETPBCOPY_H
#define KCC_WAVECODE_WAVECODETPBCOPY_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

//#include "compisa/inc/compisasimmemcpy.hpp"

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
#if 0
    bool Copied(const FileRange& tongaRange) const;
#endif

private:
    const WaveCodeSbAtomLoadKelf*   m_WaveCodeSbAtomLoadKelf = nullptr;
    std::set<TransferRange> m_NotCopiedFileToSbufAddress;
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

