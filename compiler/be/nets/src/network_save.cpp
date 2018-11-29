#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "utils/inc/asserter.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"
#include "arch/inc/arch.hpp"

#include "dma/inc/dmaqueue.hpp"


#include "nets/inc/network_save.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
#include "wave/inc/tensorscalarconstwaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "serialize/inc/serwaveop.hpp"

namespace kcc {
namespace nets {


//--------------------------------------------------------
template<>
void
Network::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    archive(cereal::make_nvp(NetKey_NetName, m_Name));
    archive(cereal::make_nvp(NetKey_DataType,
                            std::string(m_DataType->gName())));
    archive(cereal::make_nvp(NetKey_GitVersion, m_GitVersion));


    //===========================================================================
    std::vector<serialize::SerWaveOp> serWaveOps(m_WaveOps.size());
    for (unsigned waveOpIdx = 0; waveOpIdx < m_WaveOps.size(); ++waveOpIdx) {
        serialize::SerWaveOp& serWaveOp(serWaveOps[waveOpIdx]);
        wave::WaveOp* waveOp = m_WaveOps[waveOpIdx];
        serWaveOp.m_WaveOpName = waveOp->gName();
        serWaveOp.m_LayerName = waveOp->gLayerName();

        for (auto prevWaveEdge : waveOp->gPrevWaveEdges()) {
            auto prevWaveOp = prevWaveEdge->gFromOp();
            serWaveOp.addPreviousWaveOp(prevWaveOp->gName());
            if (m_UseSem && prevWaveOp->qSbAtomWaveOp()) {
                    const char* buf = "NOSEM";
                    kcc_int32 trigOrd = -1;
                    auto sbAtomWop = dynamic_cast<wave::SbAtomWaveOp*>(prevWaveOp);
                    if (auto dmaQue = sbAtomWop->gDmaQueue()) {
                        buf = dmaQue->gName().c_str();
                        trigOrd = sbAtomWop->gTriggerOrd();
                    }
                    serWaveOp.addPreviousSemaphoreSync(buf, trigOrd);
            } else {
                serWaveOp.addPreviousEventSync(prevWaveEdge->gWaitEventMode(),
                                          prevWaveEdge->gEventId(),
                                          prevWaveEdge->gSetEventMode());
            }
        }
        serWaveOp.m_Order = waveOp->gOrder();

        if (auto sbatomWaveOp = dynamic_cast<const wave::SbAtomWaveOp*>(waveOp)) {
            m_Save->saveSbAtom(sbatomWaveOp, serWaveOp);
            continue;
        }

        if (const auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {

            m_Save->saveMatmul(matmulWaveOp, serWaveOp);
            continue;
        }
        if (const auto poolWaveOp = dynamic_cast<wave::PoolWaveOp*>(waveOp)) {
            m_Save->savePool(poolWaveOp, serWaveOp);
            continue;
        }
        if (const auto activationWaveOp = dynamic_cast<const wave::ActivationWaveOp*>(waveOp)) {
            m_Save->saveActivation(activationWaveOp, serWaveOp);
            continue;
        }
        if (const auto clipByValueWaveOp = dynamic_cast<const wave::ClipByValueWaveOp*>(waveOp)) {
            m_Save->saveClipByValue(clipByValueWaveOp, serWaveOp);
            continue;
        }
        if (const auto tensorTensorWaveOp = dynamic_cast<const wave::TensorTensorWaveOp*>(waveOp)) {
            m_Save->saveTensorTensor(tensorTensorWaveOp, serWaveOp);
            continue;
        }
        if (const auto tensorScalarConstWaveOp = dynamic_cast<const wave::TensorScalarConstWaveOp*>(waveOp)) {
            m_Save->saveTensorScalarConst(tensorScalarConstWaveOp, serWaveOp);
            continue;
        }
        if (const auto barrierWaveOp = dynamic_cast<const wave::BarrierWaveOp*>(waveOp)) {
            m_Save->saveBarrier(barrierWaveOp, serWaveOp);
            continue;
        }
        if (const auto nopWaveOp = dynamic_cast<const wave::NopWaveOp*>(waveOp)) {
            m_Save->saveNop(nopWaveOp, serWaveOp);
            continue;
        }

        Assert(false, "Unsupported WaveOp: ", waveOp->gTypeStr());
    }
    archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
}





Network::Save::Save(const Network& network)
    : m_Network(network)
{ }





void
Network::Save::saveMatmul(const wave::MatMulWaveOp* matmulWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP matmulWaveOp
    serWaveOp.m_WaveOpType = wave::MatMulWaveOp::gTypeStrStatic();

    KCC_SERIALIZE(FmapXNum);
    KCC_SERIALIZE(FmapXStep);
    KCC_SERIALIZE(FmapYNum);
    KCC_SERIALIZE(FmapYStep);
    KCC_SERIALIZE(FmapZNum);
    KCC_SERIALIZE(FmapZStep);
    KCC_SERIALIZE(IfmapsSbAddress);
    serWaveOp.m_InDtype  = matmulWaveOp->gInDtype().gName();
    KCC_SERIALIZE(NumColumnPartitions);
    KCC_SERIALIZE(NumRowPartitions);
    serWaveOp.m_OutDtype = matmulWaveOp->gOutDtype().gName();
    KCC_SERIALIZE(PsumBankId);
    KCC_SERIALIZE(PsumBankOffset);
    KCC_SERIALIZE(PsumXNum);
    KCC_SERIALIZE(PsumXStep);
    KCC_SERIALIZE(PsumYNum);
    KCC_SERIALIZE(PsumYStep);
    KCC_SERIALIZE(PsumZNum);
    KCC_SERIALIZE(PsumZStep);
    serWaveOp.m_StartTensorCalc = matmulWaveOp->qStartTensorCalc();
    serWaveOp.m_StopTensorCalc = matmulWaveOp->qStopTensorCalc();
    // waveop name
    // waveop type
    KCC_SERIALIZE(WeightsSbAddress);

    KCC_SERIALIZE(IfmapReplicationNumRows);
    KCC_SERIALIZE(IfmapReplicationResolution);
    KCC_SERIALIZE(IfmapReplicationShiftAmnt);

    if (kcc::utils::DataTypeId::Uint8 == matmulWaveOp->gInDtype().gDataTypeId() ||
        kcc::utils::DataTypeId::Uint16 == matmulWaveOp->gInDtype().gDataTypeId()) {
        KCC_SERIALIZE(QuantOffsetIfmaps);
        KCC_SERIALIZE(QuantOffsetWeights);
    }
#undef WAVE_OP
} // Network::Save::saveMatmul




void
Network::Save::savePool(const wave::PoolWaveOp* poolWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP poolWaveOp
    serWaveOp.m_WaveOpType = wave::PoolWaveOp::gTypeStrStatic();

    KCC_SERIALIZE(NumPartitions);
    KCC_SERIALIZE(PoolFrequency);
    serWaveOp.m_PoolFunc = utils::poolType2Str(poolWaveOp->gPoolFunc());

    saveSrc(WAVE_OP, serWaveOp, Dims::XYZ);
    KCC_SERIALIZE(SrcWNum);
    KCC_SERIALIZE(SrcWStep);
    saveDst(WAVE_OP, serWaveOp, Dims::XYZ);

    for (unsigned int i = 0; i < poolWaveOp->gTileId().size(); ++i) {
        serWaveOp.m_TileId[i] = poolWaveOp->gTileId()[i];
    }
    KCC_SERIALIZE(TileIdFormat);
#undef WAVE_OP
}

void
Network::Save::saveSbAtom(const wave::SbAtomWaveOp* sbatomWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP sbatomWaveOp
    serWaveOp.m_Engine = engineId2Str(WAVE_OP->gEngineId());
    KCC_SERIALIZE(SbAddress);
    KCC_SERIALIZE(StartAtMidPart);
    serWaveOp.m_DataType = DataType::dataTypeId2Str(
                              sbatomWaveOp->gDataType().gDataTypeId());
    KCC_SERIALIZE(Length);
    KCC_SERIALIZE(OffsetInFile);
    KCC_SERIALIZE(PartitionStepBytes);
    serWaveOp.m_RefFile = sbatomWaveOp->gRefFileName();
    KCC_SERIALIZE(RefFileFormat);
    const utils::TensorParams::ShapeType& refFileShape(sbatomWaveOp->gRefFileShape());
    for (unsigned int shapeIdx = 0; shapeIdx < refFileShape.size(); ++shapeIdx) {
        serWaveOp.m_RefFileShape[shapeIdx] = refFileShape[shapeIdx];
    }
#undef WAVE_OP

    if (auto sbatomLoadWaveOp = dynamic_cast<const wave::SbAtomLoadWaveOp*>(sbatomWaveOp)) {
#define WAVE_OP sbatomLoadWaveOp
        serWaveOp.m_WaveOpType = wave::SbAtomLoadWaveOp::gTypeStrStatic();
        KCC_SERIALIZE(NumPartitions);
        serWaveOp.m_ContainWeights = sbatomLoadWaveOp->qContainWeights();

        KCC_SERIALIZE(IfmapReplicationNumRows);
        KCC_SERIALIZE(IfmapReplicationResolution);
        KCC_SERIALIZE(IfmapReplicationStepBytes);

        KCC_SERIALIZE(SrcStepElem);

#undef WAVE_OP
    } else {
#define WAVE_OP sbatomsaveWaveOp
        auto sbatomsaveWaveOp = dynamic_cast<const wave::SbAtomSaveWaveOp*>(sbatomWaveOp);
        Assert(sbatomsaveWaveOp, "Wrong SbAtaom WaveOp", sbatomWaveOp->gTypeStr());
        serWaveOp.m_WaveOpType = wave::SbAtomSaveWaveOp::gTypeStrStatic();
        KCC_SERIALIZE(NumPartitions);
        serWaveOp.m_FinalLayerOfmap = sbatomsaveWaveOp->qFinalLayerOfmap();
#undef WAVE_OP
    }
} // Network::Save::saveSbAtom


void
Network::Save::saveActivation(const wave::ActivationWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP activationWaveOp
    serWaveOp.m_WaveOpType = wave::ActivationWaveOp::gTypeStrStatic();

    serWaveOp.m_ActivationFunc      = serialize::SerWaveOp::activationType2Str(activationWaveOp->gActivationFunc());
    serWaveOp.m_BiasAddEn           = activationWaveOp->qBiasAddEn();

    KCC_SERIALIZE(BiasSbAddress);
    KCC_SERIALIZE(BiasStartAtMidPart);
    KCC_SERIALIZE(Scale);
    serWaveOp.m_BiasDtype           = activationWaveOp->gBiasDtype().gName();


    saveSrc(WAVE_OP, serWaveOp, Dims::XYZ);
    saveDst(WAVE_OP, serWaveOp, Dims::XYZ);

    KCC_SERIALIZE(NumPartitions);


    const std::array<kcc_int32, 4>& tileId(activationWaveOp->gTileId());
    for (unsigned int i = 0; i < tileId.size(); ++i) {
        serWaveOp.m_TileId[i]       = tileId[i];
    }
    KCC_SERIALIZE(TileIdFormat);
#undef WAVE_OP
}


void
Network::Save::saveClipByValue(const wave::ClipByValueWaveOp* clipByValueWaveOp,
                       serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP clipByValueWaveOp
    serWaveOp.m_WaveOpType = wave::ClipByValueWaveOp::gTypeStrStatic();

    KCC_SERIALIZE(NumPartitions);
    KCC_SERIALIZE(MinValue);
    KCC_SERIALIZE(MaxValue);

    saveSrc(WAVE_OP, serWaveOp, Dims::XYZ);
    saveDst(WAVE_OP, serWaveOp, Dims::XYZ);

    const std::array<kcc_int32, 4>& tileId(clipByValueWaveOp->gTileId());
    for (unsigned int i = 0; i < tileId.size(); ++i) {
        serWaveOp.m_TileId[i]       = tileId[i];
    }
    KCC_SERIALIZE(TileIdFormat);
#undef WAVE_OP
}



void
Network::Save::saveTensorTensor(const wave::TensorTensorWaveOp* tensorTensorWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP tensorTensorWaveOp
    serWaveOp.m_IsScalarOp = false;
    serWaveOp.m_WaveOpType = tensorTensorWaveOp->gTypeStr();

    KCC_SERIALIZE(NumPartitions);

    saveSrcAB(WAVE_OP, serWaveOp, Dims::XYZ);
    saveDst(WAVE_OP, serWaveOp, Dims::XYZ);

#undef WAVE_OP
}

void
Network::Save::saveTensorScalarConst(const wave::TensorScalarConstWaveOp* tensorScalarConstWaveOp,
                       serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP tensorScalarConstWaveOp
    serWaveOp.m_IsScalarOp = true;
    serWaveOp.m_WaveOpType = tensorScalarConstWaveOp->gTypeStr();

    KCC_SERIALIZE(NumPartitions);

    saveSrc(WAVE_OP, serWaveOp, Dims::XYZ);
    saveDst(WAVE_OP, serWaveOp, Dims::XYZ);

    if (serWaveOp.m_WaveOpType == "ScaleAdd") {
        serWaveOp.m_Scale = tensorScalarConstWaveOp->gImmVal(0);
        serWaveOp.m_Add = tensorScalarConstWaveOp->gImmVal(1);
    }
#undef WAVE_OP
}



void
Network::Save::saveBarrier(const wave::BarrierWaveOp* /*barrierWaveOp*/,
                           serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP barrierWaveOp
    serWaveOp.m_WaveOpType = wave::BarrierWaveOp::gTypeStrStatic();
#undef WAVE_OP
}

void
Network::Save::saveNop(const wave::NopWaveOp* nopWaveOp,
                           serialize::SerWaveOp& serWaveOp) const
{
#undef WAVE_OP
#define WAVE_OP nopWaveOp
    serWaveOp.m_WaveOpType = wave::NopWaveOp::gTypeStrStatic();
    serWaveOp.m_Engine = engineId2Str((WAVE_OP)->gEngineId());
#undef WAVE_OP
}

#undef KCC_SERIALIZE

}}



