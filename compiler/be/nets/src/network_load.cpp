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
#include "utils/inc/consts.hpp"
#include "arch/inc/arch.hpp"

#include "nets/inc/network_load.hpp"

#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/reciprocalwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
#include "wave/inc/tensorscalarconstwaveop.hpp"

#include "serialize/inc/serwaveop.hpp"




namespace kcc {
namespace nets {

//--------------------------------------------------------
//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(NetKey_NetName, m_Name));
    std::string dataType;
    archive(cereal::make_nvp(NetKey_DataType, dataType));

    if (dataType == DataTypeUint8::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint8>();

    } else if (dataType == DataTypeUint16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint16>();

    } else if (dataType == DataTypeFloat16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat16>();

    } else if (dataType == DataTypeBFloat16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeBFloat16>();

    } else if (dataType == DataTypeFloat32::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat32>();

    } else {
        Assert(false, "Unsupported data type ", dataType);
    }

    //===========================================================================
    if (m_UseWave) {
        std::vector<serialize::SerWaveOp> serWaveOps;
        archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
        for (unsigned i = 0; i < serWaveOps.size(); ++i) {
            const serialize::SerWaveOp& serWaveOp(serWaveOps[i]);

            wave::WaveOp* waveOp = nullptr;

            if (serWaveOp.m_WaveOpType == wave::SbAtomLoadWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadSbAtomLoad(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::SbAtomSaveWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadSbAtomSave(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::PoolWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadPool(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ReciprocalWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadReciprocal(serWaveOp);                
            } else if (serWaveOp.m_WaveOpType == wave::MatMulWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadMatMul(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ActivationWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadActivation(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ClipByValueWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadClipByValue(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrScaleAddStatic()) {
                waveOp = m_Load->loadScaleAdd(serWaveOp);

            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMultiplyStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Mult;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrAddStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Add;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrSubStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Sub;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrResAddStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Add;
                waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMinimum()) {
                waveOp = m_Load->loadMinimum(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMaximum()) {
                waveOp = m_Load->loadMaximum(serWaveOp);

            } else {
                Assert(false, "Wrong WaveOp type during deserialization: ", serWaveOp.m_WaveOpType);
            }

            m_WaveOps.push_back(waveOp);
            Assert(m_Name2WaveOp.find(waveOp->gName()) == m_Name2WaveOp.end(),
                   "Waveop ", waveOp->gName(), " already exists");
            m_Name2WaveOp[waveOp->gName()] = waveOp;
        }
    }
}




Network::Load::Load(Network& network)
    : m_Network(network)
{
}



wave::SbAtomLoadWaveOp*
Network::Load::loadSbAtomLoad(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS sbatomLoadParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomLoadWaveOp::Params sbatomLoadParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomLoadParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(StartAtMidPart);
    sbatomLoadParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType.c_str());
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomLoadParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomLoadParams.m_RefFileShape.size(); ++i) {
        sbatomLoadParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(ContainWeights);

    KCC_UNSERIALIZE(IfmapReplicationNumRows);
    KCC_UNSERIALIZE(IfmapReplicationResolution);
    KCC_UNSERIALIZE(IfmapReplicationStepBytes);

    KCC_UNSERIALIZE(SrcStepElem);

    auto waveOp = new wave::SbAtomLoadWaveOp(sbatomLoadParams, prevWaveOps);
    Assert(waveOp && waveOp->gName() == sbatomLoadParams.m_WaveOpName,
           "Wrong wave op name: should be ", sbatomLoadParams.m_WaveOpName,
           ", it is ", waveOp->gName());
    return waveOp;
#undef PARAMS
}


wave::SbAtomSaveWaveOp*
Network::Load::loadSbAtomSave(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS sbatomsaveParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomsaveParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(StartAtMidPart);
    sbatomsaveParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType.c_str());
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomsaveParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomsaveParams.m_RefFileShape.size(); ++i) {
        sbatomsaveParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(FinalLayerOfmap);

    auto waveOp = new wave::SbAtomSaveWaveOp(sbatomsaveParams, prevWaveOps);
    Assert(waveOp && waveOp->gName() == sbatomsaveParams.m_WaveOpName,
           "Wrong wave op name: should be ", sbatomsaveParams.m_WaveOpName,
           ", it is ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::PoolWaveOp*
Network::Load::loadPool(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS poolParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::PoolWaveOp::Params poolParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    KCC_UNSERIALIZE(SrcWNum);
    KCC_UNSERIALIZE(SrcWStep);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    poolParams.m_InDtypeId  = DataType::dataTypeStr2Id(serWaveOp.m_InDtype.c_str());

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(PoolFrequency);
    poolParams.m_PoolFunc  = utils::poolTypeStr2Id(serWaveOp.m_PoolFunc);

    Assert(poolParams.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", poolParams.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        poolParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    poolParams.m_TileIdFormat           = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::PoolWaveOp(poolParams, prevWaveOps);
    Assert(waveOp->gName() == poolParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::ReciprocalWaveOp*
Network::Load::loadReciprocal(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS reciprocalParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ReciprocalWaveOp::Params reciprocalParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    reciprocalParams.m_InDtypeId  = DataType::dataTypeStr2Id(serWaveOp.m_InDtype.c_str());

    KCC_UNSERIALIZE(NumPartitions);

    Assert(reciprocalParams.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", reciprocalParams.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        reciprocalParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    reciprocalParams.m_TileIdFormat = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::ReciprocalWaveOp(reciprocalParams, prevWaveOps);
    Assert(waveOp->gName() == reciprocalParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::MatMulWaveOp*
Network::Load::loadMatMul(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS matmulParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::MatMulWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(FmapXNum);
    KCC_UNSERIALIZE(FmapXStep);
    KCC_UNSERIALIZE(FmapYNum);
    KCC_UNSERIALIZE(FmapYStep);
    KCC_UNSERIALIZE(FmapZNum);
    KCC_UNSERIALIZE(FmapZStep);
    KCC_UNSERIALIZE(IfmapsSbAddress);
    PARAMS.m_InDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InDtype.c_str());
    KCC_UNSERIALIZE(NumColumnPartitions);
    KCC_UNSERIALIZE(NumRowPartitions);
    PARAMS.m_OutDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype.c_str());
    KCC_UNSERIALIZE(PsumBankId);
    KCC_UNSERIALIZE(PsumBankOffset);
    KCC_UNSERIALIZE(PsumXNum);
    KCC_UNSERIALIZE(PsumXStep);
    KCC_UNSERIALIZE(PsumYNum);
    KCC_UNSERIALIZE(PsumYStep);
    KCC_UNSERIALIZE(PsumZNum);
    KCC_UNSERIALIZE(PsumZStep);

    KCC_UNSERIALIZE(StartTensorCalc);
    KCC_UNSERIALIZE(StopTensorCalc);

    // waveop name
    // waveop type
    KCC_UNSERIALIZE(WeightsSbAddress);

    KCC_UNSERIALIZE(IfmapReplicationNumRows);
    KCC_UNSERIALIZE(IfmapReplicationResolution);
    KCC_UNSERIALIZE(IfmapReplicationShiftAmnt);

    if (utils::DataType::qNeedsQuantization(PARAMS.m_InDtypeId)) {
        KCC_UNSERIALIZE(QuantOffsetIfmaps);
        KCC_UNSERIALIZE(QuantOffsetWeights);
    }

    auto waveOp = new wave::MatMulWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
} // Network::Load::loadMatMul

wave::ActivationWaveOp*
Network::Load::loadActivation(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS activationParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ActivationWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    PARAMS.m_ActivationFunc   = serialize::SerWaveOp::str2ActivationFunc(serWaveOp.m_ActivationFunc);

    PARAMS.m_BiasDtypeId      = DataType::dataTypeStr2Id(serWaveOp.m_BiasDtype.c_str());
    KCC_UNSERIALIZE(BiasAddEn);
    KCC_UNSERIALIZE(BiasSbAddress);
    KCC_UNSERIALIZE(BiasStartAtMidPart);
    KCC_UNSERIALIZE(Scale);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(NumPartitions);


    Assert(PARAMS.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", PARAMS.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        PARAMS.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ActivationWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::ClipByValueWaveOp*
Network::Load::loadClipByValue(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS clipByValueParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ClipByValueWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(MinValue);
    KCC_UNSERIALIZE(MaxValue);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    Assert(PARAMS.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", PARAMS.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        PARAMS.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ClipByValueWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}




wave::TensorWaveOp*
Network::Load::loadMinimum(const serialize::SerWaveOp& serWaveOp)
{
    const TensorAluOpType aluOp = TensorAluOpType::Min;
    wave::TensorWaveOp* waveOp = nullptr;
    if (serWaveOp.m_IsScalarOp) {
        waveOp = loadTensorScalarConst(serWaveOp, aluOp);
    } else {
        waveOp = loadTensorTensor(serWaveOp, aluOp);
    }
    return waveOp;
}

wave::TensorWaveOp*
Network::Load::loadMaximum(const serialize::SerWaveOp& serWaveOp)
{
    const TensorAluOpType aluOp = TensorAluOpType::Max;
    wave::TensorWaveOp* waveOp = nullptr;
    if (serWaveOp.m_IsScalarOp) {
        waveOp = loadTensorScalarConst(serWaveOp, aluOp);
    } else {
        waveOp = loadTensorTensor(serWaveOp, aluOp);
    }
    return waveOp;
}

wave::TensorScalarConstWaveOp*
Network::Load::loadScaleAdd(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS tensorScalarConstParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorScalarConstWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(NumPartitions);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    if (serWaveOp.m_WaveOpType == wave::TensorScalarConstWaveOp::gTypeStrScaleAddStatic()) {
        // y = aluOp[1] * (x + aluOp[0])
        PARAMS.m_AluOp[0] = TensorAluOpType::Mult; 
        PARAMS.m_AluOp[1] = TensorAluOpType::Add; 
        PARAMS.m_ImmVal[0] = serWaveOp.m_Scale;
        PARAMS.m_ImmVal[1] = serWaveOp.m_Add;
    } else {
        Assert(false, "Supported ALU ops are: Add, Mult");
    }

    auto waveOp = new wave::TensorScalarConstWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}


wave::TensorTensorWaveOp*
Network::Load::loadTensorTensor(const serialize::SerWaveOp& serWaveOp, TensorAluOpType aluOp)
{
#undef PARAMS
#define PARAMS tensorTensorParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorTensorWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    PARAMS.m_InADtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InADtype.c_str());
    PARAMS.m_InBDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InBDtype.c_str());
    PARAMS.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype.c_str());
    KCC_UNSERIALIZE(NumPartitions);

    loadSrcAB(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    auto waveOp = new wave::TensorTensorWaveOp(aluOp, PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::TensorScalarConstWaveOp*
Network::Load::loadTensorScalarConst(const serialize::SerWaveOp& serWaveOp, TensorAluOpType aluOp)
{
#undef PARAMS
#define PARAMS tensorScalarAddParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorScalarConstWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, tensorScalarAddParams);

    PARAMS.m_InDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InDtype.c_str());
    PARAMS.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype.c_str());
    KCC_UNSERIALIZE(NumPartitions);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    switch (aluOp) {
    case TensorAluOpType::Add:
    case TensorAluOpType::Sub:
    case TensorAluOpType::Mult:
    case TensorAluOpType::Min:
    case TensorAluOpType::Max:
        PARAMS.m_AluOp[0]  = TensorAluOpType::Bypass;
        PARAMS.m_ImmVal[0] = 0.0;
        PARAMS.m_AluOp[1]  = aluOp;
        PARAMS.m_ImmVal[1] = serWaveOp.m_ScalarVal;
        break;
    default:
        Assert(false, "Supported TensorScalar ops are: Add, Sub, Mult, Minimum, Maximum: ", 
            static_cast<kcc_int32>(aluOp));
        break;
    }

    auto waveOp = new wave::TensorScalarConstWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}



/* in
 * template<>
 * void
 * Network::Load::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
 * {
 *      ...
 *      auto fillWaveOpParams = [this, &prevWaveOps](
 *                              const serialize::SerWaveOp& serWaveOp,
 *                              wave::WaveOp::Params& waveOpParams) -> void
 *      ...
 * }
 */

void
Network::Load::fillWaveOpParams(const serialize::SerWaveOp& serWaveOp,
                     std::vector<wave::WaveOp*>& prevWaveOps,
                     wave::WaveOp::Params& waveOpParams)
{
    waveOpParams.m_WaveOpName   = serWaveOp.m_WaveOpName;
    waveOpParams.m_LayerName    = serWaveOp.m_LayerName;
    waveOpParams.m_Order        = m_Network.gWaveOps().size();
    Assert(waveOpParams.m_LayerName != "", "Missing layer name for waveop ", serWaveOp.m_WaveOpName);
    for (const auto& prevWaveOpName : serWaveOp.m_PreviousWaveOps) {
        prevWaveOps.push_back(m_Network.findWaveOp(prevWaveOpName));
    }
}




#undef KCC_UNSERIALIZE

}}


