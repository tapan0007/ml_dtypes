#pragma once

#ifndef KCC_NETS_NETWORK_LOAD_H
#define KCC_NETS_NETWORK_LOAD_H

#include "serialize/inc/serwaveop.hpp"

#include "nets/inc/network.hpp"

#undef KCC_UNSERIALIZE
#define KCC_UNSERIALIZE(X) (PARAMS).KCC_CONCAT(m_,X) = serWaveOp.KCC_CONCAT(m_,X)

namespace kcc {

namespace nets {

class Network::Load {
public:
    Load(Network& network);
    wave::SbAtomLoadWaveOp* loadSbAtomLoad(const serialize::SerWaveOp& serWaveOp);
    wave::SbAtomSaveWaveOp* loadSbAtomSave(const serialize::SerWaveOp& serWaveOp);
    wave::PoolWaveOp* loadPool(const serialize::SerWaveOp& serWaveOp);
    wave::ReciprocalWaveOp* loadReciprocal(const serialize::SerWaveOp& serWaveOp);
    wave::MatMulWaveOp* loadMatMul(const serialize::SerWaveOp& serWaveOp);
    wave::ActivationWaveOp* loadActivation(const serialize::SerWaveOp& serWaveOp);
    wave::ClipByValueWaveOp* loadClipByValue(const serialize::SerWaveOp& serWaveOp);

    wave::TensorTensorWaveOp* loadTensorTensor(const serialize::SerWaveOp&, TensorAluOpType);
    wave::TensorScalarConstWaveOp* loadTensorScalarConst(const serialize::SerWaveOp&, TensorAluOpType);

    wave::TensorWaveOp* loadMinimum(const serialize::SerWaveOp& serWaveOp);
    wave::TensorWaveOp* loadMaximum(const serialize::SerWaveOp& serWaveOp);

    wave::TensorScalarConstWaveOp* loadScaleAdd(const serialize::SerWaveOp& serWaveOp);


private:
    template <typename ParamsType>
    void loadSrc(ParamsType& PARAMS, const serialize::SerWaveOp& serWaveOp, Dims dims)
    {
        PARAMS.m_InDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
        KCC_UNSERIALIZE(SrcIsPsum);
        if (serWaveOp.m_SrcIsPsum) {
            KCC_UNSERIALIZE(SrcPsumBankId);
            KCC_UNSERIALIZE(SrcPsumBankOffset);
        } else {
            KCC_UNSERIALIZE(SrcSbAddress);
            KCC_UNSERIALIZE(SrcStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_UNSERIALIZE(SrcWNum);
        //    KCC_UNSERIALIZE(SrcWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_UNSERIALIZE(SrcZNum);
            KCC_UNSERIALIZE(SrcZStep);
            // fall through!
        case Dims::XY:
            KCC_UNSERIALIZE(SrcYNum);
            KCC_UNSERIALIZE(SrcYStep);
            // fall through!
        case Dims::X:
            KCC_UNSERIALIZE(SrcXNum);
            KCC_UNSERIALIZE(SrcXStep);
            break;
        default:
            Assert(false, "Wrong dim for load Src");
        }
    }

    template <typename ParamsType>
    void loadSrcA(ParamsType& PARAMS, const serialize::SerWaveOp& serWaveOp, Dims dims)
    {
        PARAMS.m_InADtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InADtype);
        KCC_UNSERIALIZE(SrcAIsPsum);
        if (serWaveOp.m_SrcAIsPsum) {
            KCC_UNSERIALIZE(SrcAPsumBankId);
            KCC_UNSERIALIZE(SrcAPsumBankOffset);
        } else {
            KCC_UNSERIALIZE(SrcASbAddress);
            KCC_UNSERIALIZE(SrcAStartAtMidPart);
        }
        switch (dims) {
        case Dims::XYZW:
        //    KCC_UNSERIALIZE(SrcAWNum);
        //    KCC_UNSERIALIZE(SrcAWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_UNSERIALIZE(SrcAZNum);
            KCC_UNSERIALIZE(SrcAZStep);
            // fall through!
        case Dims::XY:
            KCC_UNSERIALIZE(SrcAYNum);
            KCC_UNSERIALIZE(SrcAYStep);
            // fall through!
        case Dims::X:
            KCC_UNSERIALIZE(SrcAXNum);
            KCC_UNSERIALIZE(SrcAXStep);
            break;
        default:
            Assert(false, "Wrong dim for load SrcA");
        }
    }

    template <typename ParamsType>
    void loadSrcB(ParamsType& PARAMS, const serialize::SerWaveOp& serWaveOp, Dims dims)
    {
        PARAMS.m_InBDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InBDtype);
        KCC_UNSERIALIZE(SrcBIsPsum);
        if (serWaveOp.m_SrcBIsPsum) {
            KCC_UNSERIALIZE(SrcBPsumBankId);
            KCC_UNSERIALIZE(SrcBPsumBankOffset);
        } else {
            KCC_UNSERIALIZE(SrcBSbAddress);
            KCC_UNSERIALIZE(SrcBStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_UNSERIALIZE(SrcBWNum);
        //    KCC_UNSERIALIZE(SrcBWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_UNSERIALIZE(SrcBZNum);
            KCC_UNSERIALIZE(SrcBZStep);
            // fall through!
        case Dims::XY:
            KCC_UNSERIALIZE(SrcBYNum);
            KCC_UNSERIALIZE(SrcBYStep);
            // fall through!
        case Dims::X:
            KCC_UNSERIALIZE(SrcBXNum);
            KCC_UNSERIALIZE(SrcBXStep);
            break;
        default:
            Assert(false, "Wrong dim for load SrcB");
        }
    }

    template <typename ParamsType>
    void loadSrcAB(ParamsType& params, const serialize::SerWaveOp& serWaveOp, Dims dims)
    {
        loadSrcA(params, serWaveOp, dims);
        loadSrcB(params, serWaveOp, dims);
    }

    template <typename ParamsType>
    void loadDst(ParamsType& PARAMS, const serialize::SerWaveOp& serWaveOp, Dims dims)
    {
        PARAMS.m_OutDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
        KCC_UNSERIALIZE(DstStartAtMidPart);
        KCC_UNSERIALIZE(DstIsPsum);
        if (serWaveOp.m_DstIsPsum) {
            KCC_UNSERIALIZE(DstPsumBankId);
            KCC_UNSERIALIZE(DstPsumBankOffset);
        } else {
            KCC_UNSERIALIZE(DstSbAddress);
            KCC_UNSERIALIZE(DstStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_UNSERIALIZE(DstWNum);
        //    KCC_UNSERIALIZE(DstWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_UNSERIALIZE(DstZNum);
            KCC_UNSERIALIZE(DstZStep);
            // fall through!
        case Dims::XY:
            KCC_UNSERIALIZE(DstYNum);
            KCC_UNSERIALIZE(DstYStep);
            // fall through!
        case Dims::X:
            KCC_UNSERIALIZE(DstXNum);
            KCC_UNSERIALIZE(DstXStep);
            break;
        default:
            Assert(false, "Wrong dim for load SrcA");
        }
    }

    void
    fillWaveOpParams(const serialize::SerWaveOp& serWaveOp,
                     std::vector<wave::WaveOp*>& prevWaveOps,
                     wave::WaveOp::Params& waveOpParams);
private:
    Network& m_Network;
}; // class Network::Load

}}

#endif

