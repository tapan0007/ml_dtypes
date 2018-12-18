#pragma once

#ifndef KCC_WAVE_REGLOADWAVEOP_H
#define KCC_WAVE_REGLOADWAVEOP_H


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


class RegLoadWaveOp : public RegWaveOp {
private:
    using BaseClass = RegWaveOp;
public:
    class Params;
public:
    RegLoadWaveOp(const RegLoadWaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    RegLoadWaveOp() = delete;

public:
    const DataType& gInDtype () const {
        return m_InDtype;
    }
    bool qSrcIsPsum() const {
        return false;
    }
    kcc_int64 gSrcSbAddress () const {
        return m_SrcSbAddress;
    }
    bool gSrcStartAtMidPart () const {
        return m_SrcStartAtMidPart;
    }
    kcc_int32 gSrcXNum () const {
        return m_SrcXNum;
    }
    kcc_int32 gSrcXStep () const {
        return m_SrcXStep;
    }
    kcc_int32 gSrcYNum () const {
        return m_SrcYNum;
    }
    kcc_int32 gSrcYStep () const {
        return m_SrcYStep;
    }
    kcc_int32 gSrcZNum () const {
        return m_SrcZNum;
    }
    kcc_int32 gSrcZStep () const {
        return m_SrcZStep;
    }
    bool qRegLoadWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::RegLoad;
    }

private:
    const DataType&             m_InDtype;
    kcc_int64                   m_SrcSbAddress          = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = -1;
    kcc_int32                   m_SrcZStep              = -1;

    const bool                  m_SrcIsPsum             = false;
    bool                        m_SrcStartAtMidPart     = false;
}; // class RegLoadWaveOp : public WaveOp






class RegLoadWaveOp::Params : public RegLoadWaveOp::BaseClass::Params {
public:
    bool verify() const;
public:
    DataTypeId                  m_InDtypeId       = DataTypeId::None;
    kcc_int32                   m_SrcSbAddress  = -1;

    kcc_int32                   m_SrcXNum       = -1;
    kcc_int32                   m_SrcXStep      = -1;
    kcc_int32                   m_SrcYNum       = -1;
    kcc_int32                   m_SrcYStep      = -1;
    kcc_int32                   m_SrcZNum       = -1;
    kcc_int32                   m_SrcZStep      = -1;

    const bool                  m_SrcIsPsum     = false;
    bool                        m_SrcStartAtMidPart;
};


}}

#endif


