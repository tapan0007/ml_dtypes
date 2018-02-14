#pragma once

#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include <string>

#include "uarch_cfg.hpp"
#include "utils/inc/types.hpp"

#include "arch/inc/pearray.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
class Arch {
private:
    // This is the only row/column configuration for float16 and int16 data types
    // For int8 it is possible to configure pe-array as nrow=256,ncol=64 or as nrow=128,ncol=128
    enum : kcc_int32 {
        numberPeRows            = utils::power2(ROW_BITS),
        numberPeColumns         = utils::power2(COLUMN_BITS),

        numberPsumBanks         = utils::power2(BANKS_PER_COLUMN_BITS),
        numberPsumBankEntries   = utils::power2(PSUM_NUM_ENTRY_BITS),
    };
    static_assert(numberPeRows == 128, "Number PE rows not 128"); // temporary
    static_assert(numberPeColumns == 64, "Number PE columns not 64"); // temporary
    static_assert(numberPsumBanks == 4, "Number PSUM banks not 4"); // temporary
    static_assert(numberPsumBankEntries == 256, "Number PSUM entries not 256"); // temporary

    enum : kcc_int64 {
        stateBuffersSizeInBytes = 8 * 1024 * 1024, // No macro that represents exact size of partition
        sbPartitionSizeInBytes  = stateBuffersSizeInBytes  / numberPeRows,
    };
    static_assert(sbPartitionSizeInBytes  * numberPeRows == stateBuffersSizeInBytes,
                  "SB size is not multiple of SB partition size");
public:

    //----------------------------------------------------------------
    Arch();

    //----------------------------------------------------------------
    const PeArray& gPeArray() const {
        return m_PeArray;
    }

    //----------------------------------------------------------------
    const StateBuffer& gStateBuffer() const {
        return m_StateBuffer;
    }

    //----------------------------------------------------------------
    const PsumBuffer& gPsumBuffer() const {
        return m_PsumBuffer;
    }

    //----------------------------------------------------------------
    const PoolingEng& gPoolingEng() const {
        return m_PoolingEng;
    }

    //----------------------------------------------------------------
    const ActivationEng& gActivationEng() const {
        return m_ActivationEng;
    }


    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayRows() const;

    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayColumns() const;



    //----------------------------------------------------------------
    kcc_int32 gNumberPsumBanks() const;

    //----------------------------------------------------------------
    kcc_int32 gPsumBankEntries() const;

    //----------------------------------------------------------------
    const std::string& gArchVersion() const;

private:
    PeArray        m_PeArray;
    PsumBuffer     m_PsumBuffer;
    PoolingEng     m_PoolingEng;
    ActivationEng  m_ActivationEng;
    StateBuffer    m_StateBuffer;
};

}}


#endif // KCC_ARCH_ARCH_H


