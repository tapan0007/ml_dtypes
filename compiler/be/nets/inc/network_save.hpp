#pragma once

#ifndef KCC_NETS_NETWORK_SAVE_H
#define KCC_NETS_NETWORK_SAVE_H

#include "serialize/inc/serwaveop.hpp"
#include "nets/inc/network.hpp"

#undef KCC_SERIALIZE
#define KCC_SERIALIZE(X) serWaveOp.KCC_CONCAT(m_,X) = (WAVE_OP)->KCC_CONCAT(g,X)()

namespace kcc {
namespace nets {

class Network::Save {
public:
    Save(const Network& network);

    void saveMatmul(const wave::MatMulWaveOp* matmulWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void savePool(const wave::PoolWaveOp* poolWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void saveReciprocal(const wave::ReciprocalWaveOp* poolWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void saveSbAtom(const wave::SbAtomWaveOp* sbatomWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void saveActivation(const wave::ActivationWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveClipByValue(const wave::ClipByValueWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveTensorTensor(const wave::TensorTensorWaveOp* tensorTensorWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveTensorScalar(const wave::TensorScalarWaveOp* tensorScalarWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveBarrier(const wave::BarrierWaveOp* barrierWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveNop(const wave::NopWaveOp* nopWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;

private:
    template <typename WaveOp>
    void saveSrc(const WaveOp* WAVE_OP, serialize::SerWaveOp& serWaveOp, Dims dims) const
    {
        serWaveOp.m_InDtype  = WAVE_OP->gInDtype().gName();
        serWaveOp.m_SrcIsPsum = (WAVE_OP)->qSrcIsPsum();
        if (serWaveOp.m_SrcIsPsum) {
            KCC_SERIALIZE(SrcPsumBankId);
            KCC_SERIALIZE(SrcPsumBankOffset);
        } else {
            KCC_SERIALIZE(SrcSbAddress);
            KCC_SERIALIZE(SrcStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_SERIALIZE(SrcWNum);
        //    KCC_SERIALIZE(SrcWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_SERIALIZE(SrcZNum);
            KCC_SERIALIZE(SrcZStep);
            // fall through!
        case Dims::XY:
            KCC_SERIALIZE(SrcYNum);
            KCC_SERIALIZE(SrcYStep);
            // fall through!
        case Dims::X:
            KCC_SERIALIZE(SrcXNum);
            KCC_SERIALIZE(SrcXStep);
            break;
        default:
            Assert(false, "Wrong dim for save Src");
        }
    }

    template <typename WaveOp>
    void saveSrcA(const WaveOp* WAVE_OP, serialize::SerWaveOp& serWaveOp, Dims dims) const
    {
        serWaveOp.m_InADtype  = WAVE_OP->gInADtype().gName();
        serWaveOp.m_SrcAIsPsum = (WAVE_OP)->qSrcAIsPsum();
        if (serWaveOp.m_SrcAIsPsum) {
            KCC_SERIALIZE(SrcAPsumBankId);
            KCC_SERIALIZE(SrcAPsumBankOffset);
        } else {
            KCC_SERIALIZE(SrcASbAddress);
            KCC_SERIALIZE(SrcAStartAtMidPart);
        }
        switch (dims) {
        case Dims::XYZW:
        //    KCC_SERIALIZE(SrcAWNum);
        //    KCC_SERIALIZE(SrcAWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_SERIALIZE(SrcAZNum);
            KCC_SERIALIZE(SrcAZStep);
            // fall through!
        case Dims::XY:
            KCC_SERIALIZE(SrcAYNum);
            KCC_SERIALIZE(SrcAYStep);
            // fall through!
        case Dims::X:
            KCC_SERIALIZE(SrcAXNum);
            KCC_SERIALIZE(SrcAXStep);
            break;
        default:
            Assert(false, "Wrong dim for save SrcA");
        }
    }

    template <typename WaveOp>
    void saveSrcB(const WaveOp* WAVE_OP, serialize::SerWaveOp& serWaveOp, Dims dims) const
    {
        serWaveOp.m_InBDtype  = WAVE_OP->gInBDtype().gName();
        serWaveOp.m_SrcBIsPsum = (WAVE_OP)->qSrcBIsPsum();
        if (serWaveOp.m_SrcBIsPsum) {
            KCC_SERIALIZE(SrcBPsumBankId);
            KCC_SERIALIZE(SrcBPsumBankOffset);
        } else {
            KCC_SERIALIZE(SrcBSbAddress);
            KCC_SERIALIZE(SrcBStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_SERIALIZE(SrcBWNum);
        //    KCC_SERIALIZE(SrcBWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_SERIALIZE(SrcBZNum);
            KCC_SERIALIZE(SrcBZStep);
            // fall through!
        case Dims::XY:
            KCC_SERIALIZE(SrcBYNum);
            KCC_SERIALIZE(SrcBYStep);
            // fall through!
        case Dims::X:
            KCC_SERIALIZE(SrcBXNum);
            KCC_SERIALIZE(SrcBXStep);
            break;
        default:
            Assert(false, "Wrong dim for save SrcB");
        }
    }

    template <typename WaveOp>
    void saveSrcAB(const WaveOp* WAVE_OP, serialize::SerWaveOp& serWaveOp, Dims dims) const
    {
        saveSrcA(WAVE_OP, serWaveOp, dims);
        saveSrcB(WAVE_OP, serWaveOp, dims);
    }

    template <typename WaveOp>
    void saveDst(const WaveOp* WAVE_OP, serialize::SerWaveOp& serWaveOp, Dims dims) const
    {
        serWaveOp.m_OutDtype  = WAVE_OP->gOutDtype().gName();
        serWaveOp.m_DstIsPsum = (WAVE_OP)->qDstIsPsum();
        if (serWaveOp.m_DstIsPsum) {
            KCC_SERIALIZE(DstPsumBankId);
            KCC_SERIALIZE(DstPsumBankOffset);
        } else {
            KCC_SERIALIZE(DstSbAddress);
            KCC_SERIALIZE(DstStartAtMidPart);
        }
        switch (dims) {
        //case Dims::XYZW:
        //    KCC_SERIALIZE(DstWNum);
        //    KCC_SERIALIZE(DstWStep);
        //    // fall through!
        case Dims::XYZ:
            KCC_SERIALIZE(DstZNum);
            KCC_SERIALIZE(DstZStep);
            // fall through!
        case Dims::XY:
            KCC_SERIALIZE(DstYNum);
            KCC_SERIALIZE(DstYStep);
            // fall through!
        case Dims::X:
            KCC_SERIALIZE(DstXNum);
            KCC_SERIALIZE(DstXStep);
            break;
        default:
            Assert(false, "Wrong dim for save Dst");
        }
    }
private:
    const Network& m_Network;
};

}}

#endif


