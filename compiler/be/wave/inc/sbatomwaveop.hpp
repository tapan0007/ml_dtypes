#pragma once

#ifndef KCC_WAVE_SBATOMWAVEOP_H
#define KCC_WAVE_SBATOMWAVEOP_H


#include <string>
#include <array>
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
    kcc_int32 gAtomId() const {
        return m_AtomId;
    }

    kcc_int32 gAtomSize() const {
        return m_AtomSize;
    }

    kcc_int32 gBatchFoldIdx () const {
        return m_BatchFoldIdx;
    }

    const utils::DataType& gDataType() const {
        return m_DataType;
    }

    kcc_int32 gLength() const {
        return m_Length;
    }

    kcc_int64 gOffsetInFile() const {
        return m_OffsetInFile;
    }

    kcc_int64 gPartitionStepBytes() const {
        return m_PartitionStepBytes;
    }

    const std::string& gRefFileName() const {
        return m_RefFileName;
    }

    kcc_int64 gAddressInPartition(kcc_int32 offsetInPartition) const {
        kcc_int64 addressInPart = gAtomId();
        addressInPart *= gWaveAtomSize(); // offset = 0
        addressInPart += offsetInPartition;
        return addressInPart;
    }

    void rRefFileName(const std::string& refFileName) {
        m_RefFileName = refFileName;
    }

    std::string     gRefFileFormat () const {
        return m_RefFileFormat;
    }

    const std::array<kcc_int32,4>& gRefFileShape () const {
        return m_RefFileShape;
    }

    EngineId gEngineId() const override {
        return EngineId::StreamProc;
    }

protected:
    bool verify() const override;

private:
    kcc_int32       m_AtomId            = -1;
    kcc_int32       m_AtomSize            = -1;
    kcc_int32       m_BatchFoldIdx      = -1;
    const utils::DataType& m_DataType;
    kcc_int64       m_Length            = -1;
    kcc_int64       m_OffsetInFile      = -1;
    kcc_int64       m_PartitionStepBytes= -1;
    std::string     m_RefFileName       = "";
    std::string     m_RefFileFormat     = "";
    std::array<kcc_int32,4> m_RefFileShape;
};




class SbAtomWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32       m_AtomId            = -1;
    kcc_int32       m_AtomSize          = -1;
    kcc_int32       m_BatchFoldIdx      = -1;
    DataTypeId      m_DataType          = DataTypeId::None;
    kcc_int64       m_Length            = -1;
    kcc_int64       m_OffsetInFile      = -1;
    kcc_int64       m_PartitionStepBytes= -1;
    std::string     m_RefFileName       = "";
    std::string     m_RefFileFormat     = "";
    std::array<kcc_int32,4> m_RefFileShape;
};

}}


#endif



