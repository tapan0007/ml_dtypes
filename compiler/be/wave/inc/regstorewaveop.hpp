#pragma once

#ifndef KCC_WAVE_REGSTOREWAVEOP_H
#define KCC_WAVE_REGSTOREWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/regwaveop.hpp"


namespace kcc {

namespace wave {


class RegStoreWaveOp : public RegWaveOp {
private:
    using BaseClass = RegWaveOp;
public:
    class Params;
public:
    RegStoreWaveOp(const RegStoreWaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    RegStoreWaveOp() = delete;

public:
    const DataType& gOutDtype () const {
        return m_OutDtype;
    }
    bool qDstIsPsum() const {
        return false;
    }
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
    }
    bool gDstStartAtMidPart () const {
        return m_DstStartAtMidPart;
    }
    kcc_int32 gDstXNum () const {
        return m_DstXNum;
    }
    kcc_int32 gDstXStep () const {
        return m_DstXStep;
    }
    kcc_int32 gDstYNum () const {
        return m_DstYNum;
    }
    kcc_int32 gDstYStep () const {
        return m_DstYStep;
    }
    kcc_int32 gDstZNum () const {
        return m_DstZNum;
    }
    kcc_int32 gDstZStep () const {
        return m_DstZStep;
    }

    bool qRegStoreWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::RegStore;
    }

private:
    const DataType&             m_OutDtype;
    kcc_int64                   m_DstSbAddress          = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;

    bool                        m_DstStartAtMidPart     = false;

}; // class RegStoreWaveOp : public WaveOp






class RegStoreWaveOp::Params : public RegStoreWaveOp::BaseClass::Params {
public:
    bool verify() const;
public:
    DataTypeId                  m_OutDtypeId       = DataTypeId::None;
    kcc_int32                   m_DstSbAddress  = -1;

    kcc_int32                   m_DstXNum       = -1;
    kcc_int32                   m_DstXStep      = -1;
    kcc_int32                   m_DstYNum       = -1;
    kcc_int32                   m_DstYStep      = -1;
    kcc_int32                   m_DstZNum       = -1;
    kcc_int32                   m_DstZStep      = -1;

    const bool                  m_DstIsPsum     =  false;
    bool                        m_DstStartAtMidPart;
};


}}

#endif


