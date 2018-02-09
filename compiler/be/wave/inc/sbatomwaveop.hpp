#pragma once

#ifndef KCC_WAVE_SBATOMWAVEOP_H
#define KCC_WAVE_SBATOMWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/waveop.hpp"


namespace kcc {
namespace wave {


class SbAtomWaveOp : public WaveOp {
public:
    class Params;
public:
    SbAtomWaveOp(const SbAtomWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    const std::string& gRefFileName() const {
        return m_RefFileName;
    }

    void rRefFileName(const std::string& refFileName) {
        m_RefFileName = refFileName;
    }

    static std::string gTypeStr() {
        return WaveOpTypeStr_SBAtomFile;
    }

    kcc_int32 gAtomId() const {
        return m_AtomId;
    }

    kcc_int32 gLength() const {
        return m_Length;
    }

    kcc_int32 gOffsetInFile() const {
        return m_OffsetInFile;
    }

    kcc_int32 gBatchFoldIdx () const {
        return m_BatchFoldIdx;
    }

protected:
    bool verify() const override;

private:
    std::string     m_RefFileName       = "";
    kcc_int32       m_BatchFoldIdx      = -1;
    kcc_int32       m_AtomId            = -1;
    kcc_int32       m_Length            = -1;
    kcc_int32       m_OffsetInFile      = -1;
};




class SbAtomWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    std::string     m_RefFileName       = "";
    kcc_int32       m_BatchFoldIdx      = -1;
    kcc_int32       m_AtomId            = -1;
    kcc_int32       m_Length            = -1;
    kcc_int32       m_OffsetInFile      = -1;
};

}}


#endif



